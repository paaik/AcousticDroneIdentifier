
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2s_std.h"
#include "driver/gpio.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"

static const char *TAG = "4MIC_I2S";

/* I2S0 pin assignments (Pair 1 — MASTER) */
#define I2S0_BCLK_PIN   GPIO_NUM_26   /* SCK output  drives all 4 mics      */
#define I2S0_WS_PIN     GPIO_NUM_25   /* WS  output  drives all 4 mics      */
#define I2S0_DATA_PIN   GPIO_NUM_27   /* SD  input   MIC1 L + MIC2 R        */

/* I2S1 pin assignments (Pair 2 — SLAVE) */
#define I2S1_BCLK_PIN   GPIO_NUM_32   /* SCK input  — wire to GPIO26          */
#define I2S1_WS_PIN     GPIO_NUM_14   /* WS  input  — wire to GPIO25          */
#define I2S1_DATA_PIN   GPIO_NUM_33   /* SD  input  — MIC3 L + MIC4 R        */

/*  Audio configuration  */
#define SAMPLE_RATE_HZ      48000
#define DMA_BUF_COUNT       8
#define DMA_BUF_FRAMES      256

#define READ_BUF_BYTES  (DMA_BUF_FRAMES * 2 * sizeof(int32_t))

static i2s_chan_handle_t rx0 = NULL;   /* I2S0 RX — MIC1 + MIC2 (master) */
static i2s_chan_handle_t rx1 = NULL;   /* I2S1 RX — MIC3 + MIC4 (slave)  */


/* ═══════════════════════════════════════════════════════════════════════════
 * init_i2s_master()
 *
 * Initialises I2S0 as Master RX.
 * This bus generates BCLK and WS for all 4 microphones.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void init_i2s_master(void)
{
    /* STEP 1. Allocate Channel for the Master  */
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num  = DMA_BUF_COUNT;
    chan_cfg.dma_frame_num = DMA_BUF_FRAMES;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx0));

    /* STEP 2. Allocate Channel for the Master  */
    i2s_std_clk_config_t clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ);
    clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;

    /* STEP 3. Slot configuration — Philips I2S, stereo, 32-bit */
    i2s_std_slot_config_t slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
        I2S_DATA_BIT_WIDTH_32BIT,
        I2S_SLOT_MODE_STEREO
    );

    /* STEP 4. GPIO configuration */
    i2s_std_gpio_config_t gpio_cfg = {
        .bclk = I2S0_BCLK_PIN,
        .ws   = I2S0_WS_PIN,
        .dout = I2S_GPIO_UNUSED,
        .din  = I2S0_DATA_PIN,
        .invert_flags = {
            .mclk_inv = false,
            .bclk_inv = false,
            .ws_inv   = false,
        },
    };

    /* STEP 5. Combine and initialise */
    i2s_std_config_t i2s_cfg = {
        .clk_cfg  = clk_cfg,
        .slot_cfg = slot_cfg,
        .gpio_cfg = gpio_cfg,
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx0, &i2s_cfg));

    ESP_LOGI(TAG, "I2S0 MASTER ready  SCK=GPIO%d  WS=GPIO%d  SD=GPIO%d",
             I2S0_BCLK_PIN, I2S0_WS_PIN, I2S0_DATA_PIN);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * init_i2s_slave()
 *
 * Initialises I2S1 as Slave RX.
 * This bus does NOT generate clocks — it reads BCLK and WS
 * from the lines driven by I2S0 (master).
 *
 * Key difference from master init:
 *   - I2S_ROLE_SLAVE instead of I2S_ROLE_MASTER
 *   - No clk_cfg needed — slave takes clock from the wire
 * ═══════════════════════════════════════════════════════════════════════════ */
static void init_i2s_slave(void)
{
    /* 1. Channel allocation — SLAVE
*/
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_SLAVE);
    chan_cfg.dma_desc_num  = DMA_BUF_COUNT;
    chan_cfg.dma_frame_num = DMA_BUF_FRAMES;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx1));

    /* 2. Clock configuration
                   
    i2s_std_clk_config_t clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ);
    clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;

    /* 3. Slot configuration — must match master exactly */
    i2s_std_slot_config_t slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
        I2S_DATA_BIT_WIDTH_32BIT,
        I2S_SLOT_MODE_STEREO
    );

    /* 4. GPIO configuration
     *    I2S1_BCLK_PIN and I2S1_WS_PIN are physically wired to
     *    I2S0_BCLK_PIN and I2S0_WS_PIN on the PCB.
     *    The slave peripheral reads those signals as inputs.                 */
    
    i2s_std_gpio_config_t gpio_cfg = {
        .bclk = I2S1_BCLK_PIN,   /* input — receives clock from GPIO26       */
        .ws   = I2S1_WS_PIN,     /* input — receives WS    from GPIO25       */
        .dout = I2S_GPIO_UNUSED,
        .din  = I2S1_DATA_PIN,
        .invert_flags = {
            .mclk_inv = false,
            .bclk_inv = false,
            .ws_inv   = false,
        },
    };

    /* 5. Combine and initialise */
    i2s_std_config_t i2s_cfg = {
        .clk_cfg  = clk_cfg,
        .slot_cfg = slot_cfg,
        .gpio_cfg = gpio_cfg,
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx1, &i2s_cfg));

    ESP_LOGI(TAG, "I2S1 SLAVE  ready  SCK=GPIO%d  WS=GPIO%d  SD=GPIO%d",
             I2S1_BCLK_PIN, I2S1_WS_PIN, I2S1_DATA_PIN);
    ESP_LOGI(TAG, "  (slave clock sourced from GPIO%d + GPIO%d)",
             I2S0_BCLK_PIN, I2S0_WS_PIN);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * mic_read_task()

 * Continuously reads from both I2S buses and processes 4-channel audio.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void mic_read_task(void *arg)
{
    /* Allocate DMA-capable buffers */
    int32_t *buf0 = heap_caps_malloc(READ_BUF_BYTES, MALLOC_CAP_DMA);
    int32_t *buf1 = heap_caps_malloc(READ_BUF_BYTES, MALLOC_CAP_DMA);

    if (!buf0 || !buf1) {
        ESP_LOGE(TAG, "Failed to allocate DMA buffers");
        vTaskDelete(NULL);
        return;
    }

    /* Enable MASTER first — this starts the clock on the wire.
     * The slave must see a valid clock before it is enabled.               */
    ESP_ERROR_CHECK(i2s_channel_enable(rx0));
    ESP_LOGI(TAG, "I2S0 master enabled — clock now active on GPIO%d/GPIO%d",
             I2S0_BCLK_PIN, I2S0_WS_PIN);

    /* Small delay to let the clock stabilise before slave locks on */
    vTaskDelay(pdMS_TO_TICKS(10));

    /* Enable SLAVE second — it will lock onto the running clock */
    ESP_ERROR_CHECK(i2s_channel_enable(rx1));
    ESP_LOGI(TAG, "I2S1 slave  enabled — locked to master clock");

    ESP_LOGI(TAG, "Recording started — %d Hz, BCLK %.3f MHz — ALL 4 MICS SYNCHRONISED",
             SAMPLE_RATE_HZ, (SAMPLE_RATE_HZ * 64) / 1000000.0f);

    size_t bytes_read0 = 0;   /* separate variables — fixes the original bug */
    size_t bytes_read1 = 0;

    while (1) {
        /* Read from both buses */
        ESP_ERROR_CHECK(
            i2s_channel_read(rx0, buf0, READ_BUF_BYTES, &bytes_read0, portMAX_DELAY)
        );
        ESP_ERROR_CHECK(
            i2s_channel_read(rx1, buf1, READ_BUF_BYTES, &bytes_read1, portMAX_DELAY)
        );

        /* Use the smaller of the two in case one returned fewer bytes */
        int frames = (int)(
            (bytes_read0 < bytes_read1 ? bytes_read0 : bytes_read1)
            / (2 * sizeof(int32_t))
        );

        /* ── De-interleave and compute RMS per channel ────────────────────── */
        int64_t sum_sq[4] = {0, 0, 0, 0};

        for (int i = 0; i < frames; i++) {
            int32_t mic1 = buf0[i * 2 + 0] >> 8;
            int32_t mic2 = buf0[i * 2 + 1] >> 8;
            int32_t mic3 = buf1[i * 2 + 0] >> 8;
            int32_t mic4 = buf1[i * 2 + 1] >> 8;

            sum_sq[0] += (int64_t)mic1 * mic1;
            sum_sq[1] += (int64_t)mic2 * mic2;
            sum_sq[2] += (int64_t)mic3 * mic3;
            sum_sq[3] += (int64_t)mic4 * mic4;
        }

        /* Calculate and print RMS for each mic */
        for (int c = 0; c < 4; c++) {
            int32_t rms = (int32_t)sqrtf((float)sum_sq[c] / (float)frames);
            ESP_LOGI(TAG, "Mic%d RMS: %6ld", c + 1, (long)rms);
        }

        vTaskDelay(pdMS_TO_TICKS(200));
    }

    free(buf0);
    free(buf1);
    vTaskDelete(NULL);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * app_main()
 * ═══════════════════════════════════════════════════════════════════════════ */
void app_main(void)
{
    ESP_LOGI(TAG, "=== 4x DMM-4026-B-I2S — Master/Slave Synchronised ===");
    ESP_LOGI(TAG, "Master: I2S0 GPIO26/25/27 | Slave: I2S1 GPIO32/14/33");
    ESP_LOGI(TAG, "PCB bridges required: GPIO26->GPIO32  GPIO25->GPIO14");

    /* Initialise master first, then slave */
    init_i2s_master();
    init_i2s_slave();

    ESP_LOGI(TAG, "Both channels initialised. Launching audio task...");

    xTaskCreatePinnedToCore(
        mic_read_task,
        "mic_read",
        8192,
        NULL,
        5,
        NULL,
        1
    );
}
