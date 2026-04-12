// we generate mic_sck by dividing the 12MHz clock. The division factor is set by MIC_DIV.
module i2s_mic_timing_gen (
    input  wire clk_12m,
    output reg        mic_sck       = 1'b0,
    output reg        mic_ws        = 1'b0,
    output reg        mic_bclk_tick = 1'b0,
    output reg [5:0]  mic_bit_cnt   = 6'd0,
    output reg        mic_word_done = 1'b0
);

    parameter integer MIC_DIV       = 1;
    parameter integer MIC_WORD_BITS = 32;

    reg [15:0] mic_div_cnt  = 16'd0;
    reg        mic_sck_prev = 1'b0;

    // we generate mic_sck by dividing the 12MHz clock. The division factor is set by MIC_DIV.
    always @(posedge clk_12m) begin
        mic_sck_prev <= mic_sck;    
        if (mic_div_cnt == MIC_DIV) begin
            mic_div_cnt <= 0;
            mic_sck     <= ~mic_sck; // we toggle mic_sck at the end of the count, so that the first tick is at count=0 
        end else begin
            mic_div_cnt <= mic_div_cnt + 1;
        end
    end

    // mic_bclk_tick is a one-cycle pulse that indicates the falling edge of mic_sck. We use the falling edge to capture data.
    always @(posedge clk_12m) begin
        mic_bclk_tick <= (mic_sck_prev & ~mic_sck);
    end


    always @(posedge clk_12m) begin
        mic_word_done <= 1'b0; // default pulse only

        if (mic_bclk_tick) begin
            if (mic_bit_cnt == MIC_WORD_BITS - 1)
                mic_bit_cnt <= 0;
            else
                mic_bit_cnt <= mic_bit_cnt + 1;

            if (mic_bit_cnt == MIC_WORD_BITS - 1)
                mic_word_done <= 1'b1;

            if (mic_bit_cnt == MIC_WORD_BITS - 2)
                mic_ws <= ~mic_ws;
        end
    end

endmodule
