/*
Copyright 2024 Tomoaki Itoh
This software is released under the MIT License, see LICENSE.txt.
//*/

#include "unit_revfx.h"
#include <climits>
#include "utils/buffer_ops.h"
#include "LCWReverb.h"
#include "LCWReverbParam.h"

enum {
    TIME = 0U,
    DEPTH,
    MIX,
    NUM_PARAMS
};

#define param_10bit_to_6bit(val) (val >> 4) // 0..0x3FF -> 0..0x3F

static struct {
    int32_t time = 0;
    int32_t depth = 0;
    float mix = 0.f;
} s_param;

static unit_runtime_desc_t runtime_desc;
static float *reverb_ram_pre_buffer;
static float *reverb_ram_comb_buffer;
static float *reverb_ram_ap_buffer;

static LCWReverbBlock reverbBlock;

static fast_inline float softclip(float x)
{
    const float pre = 1.f/4.f;
    const float post = 4.f;

    return fx_softclipf(1.f/3.f, x * pre) * post;
}

// ---- Callbacks exposed to runtime ----------------------------------------------

__unit_callback int8_t unit_init(const unit_runtime_desc_t * desc) {
    // (953) // = 48000 * 0.020
    // (331) // = 48000 * 0.0068
    // (71)  // = 48000 * 0.0015
    // (241) // = 48000 * 0.005
    // (81)  // = 48000 * 0.0017
    // (23)  // = 48000 * 0.0005
    const int32_t apDelay[] = {
        953, 241,
        81,
        23
    };

    if (!desc)
        return k_unit_err_undef;

    if (desc->target != unit_header.target)
        return k_unit_err_target;

    if (!UNIT_API_IS_COMPAT(desc->api))
        return k_unit_err_api_version;

    if (desc->samplerate != 48000)
        return k_unit_err_samplerate;

    if (desc->input_channels != 2 || desc->output_channels != 2)
        return k_unit_err_geometry;

    if (!desc->hooks.sdram_alloc)
        return k_unit_err_memory;

    reverb_ram_pre_buffer = (float *)desc->hooks.sdram_alloc(LCW_REVERB_PRE_SIZE * sizeof(float));
    if (!reverb_ram_pre_buffer)
        return k_unit_err_memory;

    reverb_ram_comb_buffer = (float *)desc->hooks.sdram_alloc(LCW_REVERB_COMB_BUFFER_TOTAL * sizeof(float));
    if (!reverb_ram_comb_buffer)
        return k_unit_err_memory;

    reverb_ram_ap_buffer = (float *)desc->hooks.sdram_alloc(LCW_REVERB_AP_BUFFER_TOTAL * sizeof(float));
    if (!reverb_ram_ap_buffer)
        return k_unit_err_memory;

    buf_clr_f32(reverb_ram_pre_buffer, LCW_REVERB_PRE_BUFFER_TOTAL);
    buf_clr_f32(reverb_ram_comb_buffer, LCW_REVERB_COMB_BUFFER_TOTAL);
    buf_clr_f32(reverb_ram_ap_buffer, LCW_REVERB_AP_BUFFER_TOTAL);

    runtime_desc = *desc;

    // set default values
    s_param.time = 0;
    s_param.depth = 0;
    s_param.mix = 0.5f;

    LCWInitPreBuffer(&reverbBlock, reverb_ram_pre_buffer);
    LCWInitCombBuffer(&reverbBlock, reverb_ram_comb_buffer);
    LCWInitApBuffer(&reverbBlock, reverb_ram_ap_buffer);

    // pre-delay
    reverbBlock.preDelaySize = (48000 * 30) / 1000;

    for (int32_t i=0; i<LCW_REVERB_COMB_MAX; i++) {
        reverbBlock.combFbGain[i] = 0.f;
        reverbBlock.combDelaySize[i] = lcwCombDelaySize[i];

        LCWFilterIir1 *p = &(reverbBlock.combLpf[i]);
        const float *param = lcwCombFilterParams[i];
        p->b0 = param[0];
        p->b1 = param[1];
        p->a1 = param[2];
        p->z1 = 0.f;
    }

    for (int32_t i=0; i<LCW_REVERB_AP_MAX; i++) {
        reverbBlock.apFbGain[i] = 0.7f;
        reverbBlock.apDelaySize[i] = apDelay[i];
    }

    {
        LCWFilterIir2 *p = &(reverbBlock.lpf);
        const float *param = lcwInputFilterParams[0];
        p->b0 = param[0];
        p->b1 = param[1];
        p->b2 = param[2];
        p->a1 = param[3];
        p->a2 = param[4];
        p->z1 = p->z2 = 0.f;
    }

    {
        LCWFilterIir2 *p = &(reverbBlock.hpf);
        const float *param = lcwInputFilterParams[1];
        p->b0 = param[0];
        p->b1 = param[1];
        p->b2 = param[2];
        p->a1 = param[3];
        p->a2 = param[4];
        p->z1 = p->z2 = 0.f;
    }

    return k_unit_err_none;
}

__unit_callback void unit_teardown() {
    reverb_ram_pre_buffer = nullptr;
    reverb_ram_comb_buffer = nullptr;
    reverb_ram_ap_buffer = nullptr;
}

__unit_callback void unit_reset() {
}

__unit_callback void unit_resume() {
}

__unit_callback void unit_suspend() {
}

__unit_callback void unit_render(const float * in, float * out, uint32_t frames) {
    const float * __restrict in_p = in;
    float * __restrict out_p = out;
    const float * out_e = out_p + (frames << 1); // output_channels: 2

    // -1.0 .. +1.0 -> 0.0 .. 1.0
    const float wet = (s_param.mix + 1.f) / 2.f;
    const float dry = 1.f - wet;

    const int32_t time = param_10bit_to_6bit(s_param.time);
    for (int32_t i=0; i<LCW_REVERB_COMB_MAX; i++) {
        reverbBlock.combFbGain[i] = lcwReverbGainTable[time][i];
    }

    const float sendLevel = s_param.depth / 1023.f;

    for (; out_p != out_e; in_p += 2, out_p += 2) {
        const float xL = *(in_p + 0);
        const float xR = *(in_p + 1);

        const float tmp[] = {
            xL * sendLevel,
            xR * sendLevel
        };

        float preOut;
        LCWInputPreBuffer(&preOut, tmp, &reverbBlock);

        float combOut;
        LCWInputCombLines(&combOut, preOut, &reverbBlock);

        const float out =
            LCWInputAllPass1(combOut * .125f, &reverbBlock);
        const float yL = softclip( (dry * xL) + (wet * out) );
        const float yR = softclip( (dry * xR) + (wet * out) );

        out_p[0] = yL;
        out_p[1] = yR;
    }
}

__unit_callback void unit_set_param_value(uint8_t id, int32_t value) {
    switch (id) {
    case TIME:
        s_param.time = clipminmaxi32(0, value, 1023);
        break;
    case DEPTH:
        s_param.depth = clipminmaxi32(0, value, 1023);
        break;
    case MIX:
        // -100.0 .. 100.0 -> -1.0 .. 1.0
        value = clipminmaxi32(-1000, value, 1000);
        s_param.mix = value / 1000.f;
        break;
    default:
        break;
    }
}

__unit_callback int32_t unit_get_param_value(uint8_t id) {
    switch (id) {
    case TIME:
        return s_param.time;
        break;
    case DEPTH:
        return s_param.depth;
        break;
    case MIX:
        // -1.0 .. 1.0 -> -100.0 .. 100.0
        return (int32_t)(s_param.mix * 1000);
        break;
    default:
        break;
    }

    return INT_MIN;
}

__unit_callback const char * unit_get_param_str_value(uint8_t id, int32_t value) {
    return nullptr;
}

__unit_callback void unit_set_tempo(uint32_t tempo) {
}

__unit_callback void unit_tempo_4ppqn_tick(uint32_t counter) {
}
