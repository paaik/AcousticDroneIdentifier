module superframe_loader #(
    parameter integer WORD_BITS = 32
)(
    input  wire clk_12m,
    input  wire superframe_ready,
    input  wire [2*WORD_BITS-1:0] lane0_frame,
    input  wire [2*WORD_BITS-1:0] lane1_frame,
    input  wire [2*WORD_BITS-1:0] lane2_frame,
    input  wire [2*WORD_BITS-1:0] lane3_frame,

    output reg [8*WORD_BITS-1:0] frame_reg   = {(8*WORD_BITS){1'b0}},
    output reg                   frame_valid = 1'b0
);
// We can use a simple register to capture the superframe data when superframe_ready is asserted. The superframe consists of 4 lanes of 2 words each, so we concatenate them together into a single wide register. We also assert frame_valid to indicate that a new frame is available for processing by the serializer.
always @(posedge clk_12m) begin
    frame_valid <= 1'b0;

    if (superframe_ready) begin
        frame_reg   <= {lane0_frame, lane1_frame, lane2_frame, lane3_frame};
        frame_valid <= 1'b1;
    end
end

endmodule
