// TDM Lane Capture for 4-lane I2S Microphone Array
module tdm_lane_capture #(
    parameter integer WORD_BITS = 32
)(
    input  wire clk_12m,
    input  wire mic_bclk_tick,
    input  wire mic_word_done,
    input  wire mic_ws,

    input  wire mic_sd1,
    input  wire mic_sd2,
    input  wire mic_sd3,
    input  wire mic_sd4,

    output reg [WORD_BITS-1:0] ch0 = {WORD_BITS{1'b0}},
    output reg [WORD_BITS-1:0] ch1 = {WORD_BITS{1'b0}},
    output reg [WORD_BITS-1:0] ch2 = {WORD_BITS{1'b0}},
    output reg [WORD_BITS-1:0] ch3 = {WORD_BITS{1'b0}},
    output reg [WORD_BITS-1:0] ch4 = {WORD_BITS{1'b0}},
    output reg [WORD_BITS-1:0] ch5 = {WORD_BITS{1'b0}},
    output reg [WORD_BITS-1:0] ch6 = {WORD_BITS{1'b0}},
    output reg [WORD_BITS-1:0] ch7 = {WORD_BITS{1'b0}},

    output reg [2*WORD_BITS-1:0] lane0_frame = {(2*WORD_BITS){1'b0}},
    output reg [2*WORD_BITS-1:0] lane1_frame = {(2*WORD_BITS){1'b0}},
    output reg [2*WORD_BITS-1:0] lane2_frame = {(2*WORD_BITS){1'b0}},
    output reg [2*WORD_BITS-1:0] lane3_frame = {(2*WORD_BITS){1'b0}},
    output reg                   superframe_ready = 1'b0
);

reg [WORD_BITS-1:0] lane0_shift = {WORD_BITS{1'b0}};
reg [WORD_BITS-1:0] lane1_shift = {WORD_BITS{1'b0}};
reg [WORD_BITS-1:0] lane2_shift = {WORD_BITS{1'b0}};
reg [WORD_BITS-1:0] lane3_shift = {WORD_BITS{1'b0}};

reg word_slot_ws = 1'b0;
reg word_started = 1'b0;
// we construct the current word for each lane by shifting in the new bit from the corresponding mic_sd. The MSB of the shift register is the "current" bit, which will become the LSB of the word when it's done.
wire [WORD_BITS-1:0] lane0_word = {lane0_shift[WORD_BITS-2:0], mic_sd1};
wire [WORD_BITS-1:0] lane1_word = {lane1_shift[WORD_BITS-2:0], mic_sd2};
wire [WORD_BITS-1:0] lane2_word = {lane2_shift[WORD_BITS-2:0], mic_sd3};
wire [WORD_BITS-1:0] lane3_word = {lane3_shift[WORD_BITS-2:0], mic_sd4};
// we shift in the new bits on every mic_bclk_tick
always @(posedge clk_12m) begin
    if (mic_bclk_tick) begin
        lane0_shift <= lane0_word;
        lane1_shift <= lane1_word;
        lane2_shift <= lane2_word;
        lane3_shift <= lane3_word;
    end
end
// we use mic_ws to determine which word slot (even or odd) we're currently filling. We capture the word slot at the start of the word (first bit clock after mic_ws changes) and then use that to route the completed word to the correct channel and frame position when mic_word_done fires.
always @(posedge clk_12m) begin
    if (mic_word_done) begin
        word_started <= 1'b0;
    end else if (mic_bclk_tick && !word_started) begin
        word_slot_ws <= mic_ws;
        word_started <= 1'b1;
    end
end
// when mic_word_done fires, we know the current word is complete and can be routed to the correct channel and frame position based on word_slot_ws. We also assert superframe_ready when the second word of lane0 (ch1) is done, which indicates that all 4 lanes have a complete word and the superframe is ready to be loaded.
always @(posedge clk_12m) begin
    superframe_ready <= 1'b0;

    if (mic_word_done) begin
        if (word_slot_ws == 1'b0) begin
            // mic_word_done arrives one clk_12m cycle after the final shift.
            // At that point lane*_shift holds the completed word, while lane*_word
            // has already incorporated the next serial input bit.
            ch0 <= lane0_shift;
            ch2 <= lane1_shift;
            ch4 <= lane2_shift;
            ch6 <= lane3_shift;

            lane0_frame[2*WORD_BITS-1:WORD_BITS] <= lane0_shift;
            lane1_frame[2*WORD_BITS-1:WORD_BITS] <= lane1_shift;
            lane2_frame[2*WORD_BITS-1:WORD_BITS] <= lane2_shift;
            lane3_frame[2*WORD_BITS-1:WORD_BITS] <= lane3_shift;
        end else begin
            ch1 <= lane0_shift;
            ch3 <= lane1_shift;
            ch5 <= lane2_shift;
            ch7 <= lane3_shift;

            lane0_frame[WORD_BITS-1:0] <= lane0_shift;
            lane1_frame[WORD_BITS-1:0] <= lane1_shift;
            lane2_frame[WORD_BITS-1:0] <= lane2_shift;
            lane3_frame[WORD_BITS-1:0] <= lane3_shift;

            superframe_ready <= 1'b1;
        end
    end
end

endmodule
