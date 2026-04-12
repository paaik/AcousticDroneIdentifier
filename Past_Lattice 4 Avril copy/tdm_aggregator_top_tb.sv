`timescale 1ns/1ps

module tdm_aggregator_top_tb;

localparam int WORD_BITS  = 32;
localparam int FRAME_BITS = 8 * WORD_BITS;
localparam int ST_SYNC    = 2'd1;
localparam int ST_RUN     = 2'd2;

reg clk_12m = 1'b0;
reg pi_aln  = 1'b1;

reg mic_sd1 = 1'b0;
reg mic_sd2 = 1'b0;
reg mic_sd3 = 1'b0;
reg mic_sd4 = 1'b0;

wire mic_sck;
wire mic_ws;
wire pi_sck;
wire pi_sd;

reg [FRAME_BITS-1:0] sampled_frame;
integer checks_passed = 0;

reg [31:0] exp_ch0 [0:5];
reg [31:0] exp_ch1 [0:5];
reg [31:0] exp_ch2 [0:5];
reg [31:0] exp_ch3 [0:5];
reg [31:0] exp_ch4 [0:5];
reg [31:0] exp_ch5 [0:5];
reg [31:0] exp_ch6 [0:5];
reg [31:0] exp_ch7 [0:5];
reg [8*32-1:0] exp_label [0:5];

always #41.667 clk_12m = ~clk_12m;

tdm_aggregator_top dut (
    .clk_12m(clk_12m),
    .mic_sck(mic_sck),
    .mic_ws(mic_ws),
    .mic_sd1(mic_sd1),
    .mic_sd2(mic_sd2),
    .mic_sd3(mic_sd3),
    .mic_sd4(mic_sd4),
    .pi_aln(pi_aln),
    .pi_sck(pi_sck),
    .pi_sd(pi_sd)
);

function automatic [FRAME_BITS-1:0] pack_frame(
    input [31:0] ch0,
    input [31:0] ch1,
    input [31:0] ch2,
    input [31:0] ch3,
    input [31:0] ch4,
    input [31:0] ch5,
    input [31:0] ch6,
    input [31:0] ch7
);
begin
    pack_frame = {ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7};
end
endfunction

task automatic expect_equal_32(
    input [8*32-1:0] what,
    input [31:0] actual,
    input [31:0] expected
);
begin
    if (actual !== expected) begin
        $display("[%0t] ERROR: %0s actual=%h expected=%h", $time, what, actual, expected);
        $fatal(1);
    end
    checks_passed = checks_passed + 1;
end
endtask

task automatic expect_equal_frame(
    input [8*32-1:0] what,
    input [FRAME_BITS-1:0] actual,
    input [FRAME_BITS-1:0] expected
);
begin
    if (actual !== expected) begin
        $display("[%0t] ERROR: %0s actual=%h expected=%h", $time, what, actual, expected);
        $fatal(1);
    end
    checks_passed = checks_passed + 1;
end
endtask

task automatic wait_for_stereo_frame_start;
begin
    @(negedge mic_sck);
    while (!(mic_ws == 1'b0 && dut.u_timing_gen.mic_bit_cnt == 0)) begin
        @(negedge mic_sck);
    end
end
endtask

task automatic drive_stereo_frame(
    input [31:0] ch0,
    input [31:0] ch1,
    input [31:0] ch2,
    input [31:0] ch3,
    input [31:0] ch4,
    input [31:0] ch5,
    input [31:0] ch6,
    input [31:0] ch7
);
    integer bit_idx;
begin
    for (bit_idx = WORD_BITS - 1; bit_idx >= 0; bit_idx = bit_idx - 1) begin
        mic_sd1 = ch0[bit_idx];
        mic_sd2 = ch2[bit_idx];
        mic_sd3 = ch4[bit_idx];
        mic_sd4 = ch6[bit_idx];
        @(negedge mic_sck);
    end

    for (bit_idx = WORD_BITS - 1; bit_idx >= 0; bit_idx = bit_idx - 1) begin
        mic_sd1 = ch1[bit_idx];
        mic_sd2 = ch3[bit_idx];
        mic_sd3 = ch5[bit_idx];
        mic_sd4 = ch7[bit_idx];
        @(negedge mic_sck);
    end
end
endtask

task automatic sample_next_serial_frame(output reg [FRAME_BITS-1:0] sampled);
    integer bit_idx;
begin
    sampled = {FRAME_BITS{1'b0}};

    wait (dut.u_pi_stream_out.state == ST_RUN &&
          dut.u_pi_stream_out.ser_busy &&
          dut.u_pi_stream_out.ser_bit_cnt == 1);

    for (bit_idx = FRAME_BITS - 1; bit_idx >= 0; bit_idx = bit_idx - 1) begin
        // pi_sd is updated on the clk_12m rising edge, so sample every bit on
        // the following falling edge when it is stable.
        @(negedge clk_12m);
        sampled[bit_idx] = pi_sd;
    end
end
endtask

task automatic sample_next_marker_frame(output reg [FRAME_BITS-1:0] sampled);
    integer bit_idx;
begin
    sampled = {FRAME_BITS{1'b0}};

    wait (dut.u_pi_stream_out.state == ST_SYNC &&
          dut.u_pi_stream_out.ser_busy &&
          dut.u_pi_stream_out.ser_bit_cnt == 1);

    for (bit_idx = FRAME_BITS - 1; bit_idx >= 0; bit_idx = bit_idx - 1) begin
        @(negedge clk_12m);
        sampled[bit_idx] = pi_sd;
    end
end
endtask

task automatic check_frame(
    input integer idx
);
    reg [FRAME_BITS-1:0] expected_frame;
begin
    expected_frame = pack_frame(exp_ch0[idx], exp_ch1[idx], exp_ch2[idx], exp_ch3[idx],
                                exp_ch4[idx], exp_ch5[idx], exp_ch6[idx], exp_ch7[idx]);

    @(posedge dut.u_frame_packer.frame_valid);
    #1;

    expect_equal_32({exp_label[idx], " ch0"}, dut.u_capture.ch0, exp_ch0[idx]);
    expect_equal_32({exp_label[idx], " ch1"}, dut.u_capture.ch1, exp_ch1[idx]);
    expect_equal_32({exp_label[idx], " ch2"}, dut.u_capture.ch2, exp_ch2[idx]);
    expect_equal_32({exp_label[idx], " ch3"}, dut.u_capture.ch3, exp_ch3[idx]);
    expect_equal_32({exp_label[idx], " ch4"}, dut.u_capture.ch4, exp_ch4[idx]);
    expect_equal_32({exp_label[idx], " ch5"}, dut.u_capture.ch5, exp_ch5[idx]);
    expect_equal_32({exp_label[idx], " ch6"}, dut.u_capture.ch6, exp_ch6[idx]);
    expect_equal_32({exp_label[idx], " ch7"}, dut.u_capture.ch7, exp_ch7[idx]);
    expect_equal_frame({exp_label[idx], " frame_reg"}, dut.u_frame_packer.frame_reg, expected_frame);

    $display("[%0t] PASS: %0s capture and packing are correct", $time, exp_label[idx]);
end
endtask

task automatic check_serial_frame(
    input integer idx
);
    reg [FRAME_BITS-1:0] expected_frame;
begin
    expected_frame = pack_frame(exp_ch0[idx], exp_ch1[idx], exp_ch2[idx], exp_ch3[idx],
                                exp_ch4[idx], exp_ch5[idx], exp_ch6[idx], exp_ch7[idx]);
    sample_next_serial_frame(sampled_frame);
    expect_equal_frame({exp_label[idx], " pi_stream"}, sampled_frame, expected_frame);
    $display("[%0t] PASS: %0s serializer output is correct", $time, exp_label[idx]);
end
endtask

task automatic check_marker_frame;
    reg [FRAME_BITS-1:0] expected_frame;
begin
    expected_frame = {FRAME_BITS{1'b1}};
    sample_next_marker_frame(sampled_frame);
    expect_equal_frame("ALIGN marker", sampled_frame, expected_frame);
    $display("[%0t] PASS: ALIGN marker serializer output is correct", $time);
end
endtask

always @(posedge clk_12m) begin
    if (dut.u_pi_stream_out.overflow) begin
        $display("[%0t] ERROR: serializer overflow asserted", $time);
        $fatal(1);
    end
end

initial begin
    $dumpfile("tdm_aggregator_top_tb.vcd");
    $dumpvars(0, tdm_aggregator_top_tb);
end

initial begin
    #2_000_000;
    $display("[%0t] ERROR: watchdog timeout", $time);
    $fatal(1);
end

initial begin
    exp_label[0] = "IDLE";
    exp_ch0[0] = 32'h00000000; exp_ch1[0] = 32'h00000000; exp_ch2[0] = 32'h00000000; exp_ch3[0] = 32'h00000000;
    exp_ch4[0] = 32'h00000000; exp_ch5[0] = 32'h00000000; exp_ch6[0] = 32'h00000000; exp_ch7[0] = 32'h00000000;
    exp_label[1] = "WARMUP";
    exp_ch0[1] = 32'h00000000; exp_ch1[1] = 32'h00000000; exp_ch2[1] = 32'h00000000; exp_ch3[1] = 32'h00000000;
    exp_ch4[1] = 32'h00000000; exp_ch5[1] = 32'h00000000; exp_ch6[1] = 32'h00000000; exp_ch7[1] = 32'h00000000;
    exp_label[2] = "FRAME1";
    exp_ch0[2] = 32'hDEADBEEF; exp_ch1[2] = 32'hCAFEBABE; exp_ch2[2] = 32'h12345678; exp_ch3[2] = 32'h89ABCDEF;
    exp_ch4[2] = 32'h0BADF00D; exp_ch5[2] = 32'h13579BDF; exp_ch6[2] = 32'h55AA55AA; exp_ch7[2] = 32'hA5A5C3C3;
    exp_label[3] = "FRAME2";
    exp_ch0[3] = 32'h11112222; exp_ch1[3] = 32'h33334444; exp_ch2[3] = 32'h55667788; exp_ch3[3] = 32'h99AABBCC;
    exp_ch4[3] = 32'hD00DFEED; exp_ch5[3] = 32'h2468ACE0; exp_ch6[3] = 32'hFACEB00C; exp_ch7[3] = 32'hC001D00D;
    exp_label[4] = "FRAME3";
    exp_ch0[4] = 32'h00000000; exp_ch1[4] = 32'hFFFFFFFF; exp_ch2[4] = 32'hAAAAAAAA; exp_ch3[4] = 32'h55555555;
    exp_ch4[4] = 32'h0000FFFF; exp_ch5[4] = 32'hFFFF0000; exp_ch6[4] = 32'h13572468; exp_ch7[4] = 32'h24681357;
    exp_label[5] = "FRAME4";
    exp_ch0[5] = 32'h80000001; exp_ch1[5] = 32'h7FFFFFFE; exp_ch2[5] = 32'h01234567; exp_ch3[5] = 32'h89ABCDEF;
    exp_ch4[5] = 32'hFEDCBA98; exp_ch5[5] = 32'h76543210; exp_ch6[5] = 32'h0F0F0F0F; exp_ch7[5] = 32'hF0F0F0F0;

    repeat (8) @(posedge clk_12m);
    pi_aln = 1'b0;

    fork
        begin
            wait_for_stereo_frame_start();
            drive_stereo_frame(32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000,
                               32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000);
            drive_stereo_frame(32'hDEADBEEF, 32'hCAFEBABE, 32'h12345678, 32'h89ABCDEF,
                               32'h0BADF00D, 32'h13579BDF, 32'h55AA55AA, 32'hA5A5C3C3);
            drive_stereo_frame(32'h11112222, 32'h33334444, 32'h55667788, 32'h99AABBCC,
                               32'hD00DFEED, 32'h2468ACE0, 32'hFACEB00C, 32'hC001D00D);
            drive_stereo_frame(32'h00000000, 32'hFFFFFFFF, 32'hAAAAAAAA, 32'h55555555,
                               32'h0000FFFF, 32'hFFFF0000, 32'h13572468, 32'h24681357);
            drive_stereo_frame(32'h80000001, 32'h7FFFFFFE, 32'h01234567, 32'h89ABCDEF,
                               32'hFEDCBA98, 32'h76543210, 32'h0F0F0F0F, 32'hF0F0F0F0);
        end
        begin
            integer idx;
            for (idx = 0; idx < 6; idx = idx + 1)
                check_frame(idx);
        end
        begin
            integer idx;
            check_marker_frame();
            for (idx = 0; idx < 6; idx = idx + 1)
                check_serial_frame(idx);
        end
    join

    $display("[%0t] PASS: %0d checks completed", $time, checks_passed);
    $finish;
end

endmodule
