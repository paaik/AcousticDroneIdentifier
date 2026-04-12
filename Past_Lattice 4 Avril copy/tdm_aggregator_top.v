// =============================================================
// top0.v
// Purpose : Wire the synchronous TDM aggregator together.
//
// Architecture:
//   4 x 32-bit mic word shifters
//     -> 4 x 64-bit lane frames
//     -> single SOF-style 256-bit frame load
//     -> serial output to Pi at 12 MHz
// =============================================================

module tdm_aggregator_top (
    input  wire clk_12m,

    // Mic-side outputs
    output wire mic_sck,
    output wire mic_ws,

    // Mic-side inputs
    input  wire mic_sd1,
    input  wire mic_sd2,
    input  wire mic_sd3,
    input  wire mic_sd4,

    // Pi-side
    input  wire pi_aln,
    output wire pi_sck,
    output wire pi_sd
);

parameter integer WORD_BITS  = 32;
parameter integer FRAME_BITS = 8 * WORD_BITS;
parameter integer MIC_DIV    = 1;

wire        mic_bclk_tick;
wire [5:0]  mic_bit_cnt;
wire        mic_word_done;

wire [WORD_BITS-1:0] ch0, ch1, ch2, ch3;
wire [WORD_BITS-1:0] ch4, ch5, ch6, ch7;
wire [2*WORD_BITS-1:0] lane0_frame;
wire [2*WORD_BITS-1:0] lane1_frame;
wire [2*WORD_BITS-1:0] lane2_frame;
wire [2*WORD_BITS-1:0] lane3_frame;
wire                   superframe_ready;

wire [FRAME_BITS-1:0] frame_reg;
wire                  frame_valid;

wire overflow;
wire serializer_busy;
wire rst_n = 1'b0;

assign pi_sck = clk_12m;

i2s_mic_timing_gen #(
    .MIC_DIV(MIC_DIV),
    .MIC_WORD_BITS(WORD_BITS)
) u_timing_gen (
    .clk_12m(clk_12m),
    .mic_sck(mic_sck),
    .mic_ws(mic_ws),
    .mic_bclk_tick(mic_bclk_tick),
    .mic_bit_cnt(mic_bit_cnt),
    .mic_word_done(mic_word_done)
);

tdm_lane_capture #(
    .WORD_BITS(WORD_BITS)
) u_capture (
    .clk_12m(clk_12m),
    .mic_bclk_tick(mic_bclk_tick),
    .mic_word_done(mic_word_done),
    .mic_ws(mic_ws),
    .mic_sd1(mic_sd1),
    .mic_sd2(mic_sd2),
    .mic_sd3(mic_sd3),
    .mic_sd4(mic_sd4),
    .ch0(ch0),
    .ch1(ch1),
    .ch2(ch2),
    .ch3(ch3),
    .ch4(ch4),
    .ch5(ch5),
    .ch6(ch6),
    .ch7(ch7),
    .lane0_frame(lane0_frame),
    .lane1_frame(lane1_frame),
    .lane2_frame(lane2_frame),
    .lane3_frame(lane3_frame),
    .superframe_ready(superframe_ready)
);

superframe_loader #(
    .WORD_BITS(WORD_BITS)
) u_frame_packer (
    .clk_12m(clk_12m),
    .superframe_ready(superframe_ready),
    .lane0_frame(lane0_frame),
    .lane1_frame(lane1_frame),
    .lane2_frame(lane2_frame),
    .lane3_frame(lane3_frame),
    .frame_reg(frame_reg),
    .frame_valid(frame_valid)
);

pi_tdm_serializer #(
    .FRAME_BITS(FRAME_BITS)
) u_pi_stream_out (
    .clk_12m(clk_12m),
    .rst(rst_n),
    .pi_aln(pi_aln),
    .frame_reg(frame_reg),
    .frame_valid(frame_valid),
    .pi_sd(pi_sd),
    .overflow(overflow),
    .serializer_busy(serializer_busy)
);

endmodule
