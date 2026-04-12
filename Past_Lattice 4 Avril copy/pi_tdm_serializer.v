// pi_tdm_serializer.v
module pi_tdm_serializer #(
    parameter integer FRAME_BITS = 256
)(
    input  wire                  clk_12m,
    input  wire                  rst,
    input  wire                  pi_aln,
    input  wire [FRAME_BITS-1:0] frame_reg,
    input  wire                  frame_valid,

    output reg                   pi_sd           = 1'b0,
    output reg                   overflow        = 1'b0,
    output reg                   serializer_busy = 1'b0
);

localparam ST_STOP = 2'd0;
localparam ST_SYNC = 2'd1;
localparam ST_RUN  = 2'd2;

reg [1:0] state = ST_STOP;
reg [1:0] next_state_after_shift = ST_STOP;

reg [FRAME_BITS-1:0] ser_shift_reg = {FRAME_BITS{1'b0}};
reg [$clog2(FRAME_BITS)-1:0] ser_bit_cnt = {($clog2(FRAME_BITS)){1'b0}};
reg                          ser_busy = 1'b0;

reg [FRAME_BITS-1:0] frame_latch = {FRAME_BITS{1'b0}};
reg                  frame_pending = 1'b0;
reg                  frame_pending_clear = 1'b0;
reg                  pi_aln_d = 1'b1;

localparam [FRAME_BITS-1:0] ZERO_FRAME   = {FRAME_BITS{1'b0}};
localparam [FRAME_BITS-1:0] MARKER_FRAME = {FRAME_BITS{1'b1}};

wire pi_aln_fall = pi_aln_d && !pi_aln;
wire consume_frame_valid_now;
// We can consume frame_valid immediately if we're in ST_RUN and not currently busy or pending. If we're busy or pending, we have to wait for the next opportunity (after the current shift is done) to clear the pending flag and load the new frame.
assign consume_frame_valid_now =
    (state == ST_RUN) &&
    !pi_aln &&
    !ser_busy &&
    !frame_pending &&
    frame_valid;
// Frame loading and pending logic. We latch the incoming frame when frame_valid is asserted. If we're not currently busy or pending, we can start shifting it out immediately. If we are busy or pending, we set the pending flag and will load it as soon as we're done with the current frame.
always @(posedge clk_12m or posedge rst) begin
    if (rst) begin
        frame_latch   <= {FRAME_BITS{1'b0}};
        frame_pending <= 1'b0;
        overflow      <= 1'b0;
    end else begin
        overflow <= 1'b0;

        if (frame_pending_clear)
            frame_pending <= 1'b0;

        if (frame_valid) begin
            if (frame_pending)
                overflow <= 1'b1;

            frame_latch   <= frame_reg;
            if (!consume_frame_valid_now)
                frame_pending <= 1'b1;
        end
    end
end
// Main state machine for serialization and alignment handling
always @(posedge clk_12m or posedge rst) begin
    if (rst) begin
        state                <= ST_STOP;
        next_state_after_shift <= ST_STOP;
        ser_shift_reg        <= {FRAME_BITS{1'b0}};
        ser_bit_cnt          <= {($clog2(FRAME_BITS)){1'b0}};
        ser_busy             <= 1'b0;
        serializer_busy      <= 1'b0;
        pi_sd                <= 1'b0;
        frame_pending_clear  <= 1'b0;
        pi_aln_d             <= 1'b1;
    end else begin
        serializer_busy     <= ser_busy;
        frame_pending_clear <= 1'b0;
        pi_aln_d            <= pi_aln;
// State machine transitions and output logic
        case (state)
        // While pi_aln is high we continuously send zero-valued 256-bit frames.
        // On the falling edge we immediately switch to a single all-ones marker
        // frame so the Pi can re-align to the next normal data frame.
            ST_STOP: begin
                if (pi_aln_fall) begin
                    pi_sd                 <= 1'b1;
                    ser_shift_reg         <= {MARKER_FRAME[FRAME_BITS-2:0], 1'b0};
                    ser_bit_cnt           <= 1;
                    ser_busy              <= 1'b1;
                    next_state_after_shift <= ST_RUN;
                    state                 <= ST_SYNC;
                end else if (!ser_busy) begin
                    pi_sd           <= 1'b0;
                    ser_shift_reg   <= ZERO_FRAME;
                    ser_bit_cnt     <= 1;
                    ser_busy        <= 1'b1;
                    state           <= ST_STOP;
                end else begin
                    pi_sd         <= ser_shift_reg[FRAME_BITS-1];
                    ser_shift_reg <= {ser_shift_reg[FRAME_BITS-2:0], 1'b0};

                    if (ser_bit_cnt == FRAME_BITS - 1) begin
                        ser_bit_cnt <= {($clog2(FRAME_BITS)){1'b0}};
                        ser_busy    <= 1'b0;
                    end else begin
                        ser_bit_cnt <= ser_bit_cnt + 1'b1;
                    end
                end
            end
 // In the ST_SYNC state, we shift out exactly one all-ones alignment frame.
 // If pi_aln returns high mid-marker, we abandon the marker and go back to
 // zero-frame idle mode.
            ST_SYNC: begin
                if (pi_aln) begin
                    state      <= ST_STOP;
                    ser_busy   <= 1'b0;
                    pi_sd      <= 1'b0;
                end else if (ser_busy) begin
                    pi_sd         <= ser_shift_reg[FRAME_BITS-1];
                    ser_shift_reg <= {ser_shift_reg[FRAME_BITS-2:0], 1'b0};

                    if (ser_bit_cnt == FRAME_BITS - 1) begin
                        ser_bit_cnt <= {($clog2(FRAME_BITS)){1'b0}};
                        ser_busy    <= 1'b0;
                        state       <= next_state_after_shift;
                    end else begin
                        ser_bit_cnt <= ser_bit_cnt + 1'b1;
                    end
                end
            end
// In the ST_RUN state, we shift out the actual frame data. If pi_aln goes
// high, we abandon the current transfer and return to zero-frame idle mode.
            ST_RUN: begin
                if (pi_aln) begin
                    state    <= ST_STOP;
                    ser_busy <= 1'b0;
                    pi_sd    <= 1'b0;
                end else if (!ser_busy) begin
                    if (frame_pending) begin
                        pi_sd               <= frame_latch[FRAME_BITS-1];
                        ser_shift_reg       <= {frame_latch[FRAME_BITS-2:0], 1'b0};
                        ser_bit_cnt         <= 1;
                        ser_busy            <= 1'b1;
                        frame_pending_clear <= 1'b1;
                    end else if (frame_valid) begin
                        pi_sd         <= frame_reg[FRAME_BITS-1];
                        ser_shift_reg <= {frame_reg[FRAME_BITS-2:0], 1'b0};
                        ser_bit_cnt   <= 1;
                        ser_busy      <= 1'b1;
                    end else begin
                        pi_sd <= 1'b0;
                    end
                end else begin
                    pi_sd         <= ser_shift_reg[FRAME_BITS-1];
                    ser_shift_reg <= {ser_shift_reg[FRAME_BITS-2:0], 1'b0};

                    if (ser_bit_cnt == FRAME_BITS - 1) begin
                        ser_bit_cnt <= {($clog2(FRAME_BITS)){1'b0}};
                        ser_busy    <= 1'b0;
                    end else begin
                        ser_bit_cnt <= ser_bit_cnt + 1'b1;
                    end
                end
            end
// The default case should never happen, but if it does, we go back to a safe state (ST_STOP) and clear the busy flag and output.
            default: begin
                state    <= ST_STOP;
                ser_busy <= 1'b0;
                pi_sd    <= 1'b0;
            end
        endcase
    end
end

endmodule
