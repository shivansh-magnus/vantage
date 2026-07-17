import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, status
from app.schemas import CustomerContextInput, PriceResponse, OutcomeInput, Acknowledgement
from app.state import state
from vantage.schemas import CustomerContext


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the agent state (load from disk if snapshot exists)
    state.initialize()
    yield
    # Shutdown: Perform final snapshot save
    state.save_snapshot()


app = FastAPI(
    title="Vantage Pricing Engine API",
    description="Online-learning dynamic pricing agent served via FastAPI",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check and model status dashboard."""
    if state.agent is None:
        return {"status": "loading", "message": "Model state is initializing."}
    
    return {
        "status": "healthy",
        "agent": {
            "type": "Joint LinUCB (scaled)",
            "total_rounds": state.agent.total_rounds,
            "arm_counts": state.agent.counts.tolist(),
            "q_estimates": state.agent.q_estimates.tolist(),
        }
    }


@app.get("/price", response_model=PriceResponse)
async def get_price(
    segment: str = Query(..., description="Customer segment: 'student', 'professional', or 'default'"),
    day_type: str = Query(..., description="Day type: 'weekday' or 'weekend'"),
    competitor_price: float = Query(..., description="Continuous competitor price in dollars"),
):
    """
    Suggests a price for a given customer context.
    Generates a unique request_id to correlate with a future purchase outcome.
    """
    if segment not in ["student", "professional", "default"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid segment. Must be 'student', 'professional', or 'default'."
        )
    if day_type not in ["weekday", "weekend"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid day_type. Must be 'weekday' or 'weekend'."
        )
    if competitor_price < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Competitor price cannot be negative."
        )

    # 1. Parse into CustomerContext
    ctx = CustomerContext(
        segment=segment,
        day_type=day_type,
        competitor_price=competitor_price
    )

    # 2. Extract features and normalize competitor price for Joint LinUCB
    x = ctx.to_vector()
    x_agent = x.copy()
    x_agent[4] /= 20.0  # Feature-scale normalization

    # 3. Select arm (incorporating exploration bonus)
    # We acquire the lock to prevent select/update interleaving
    async with state.lock:
        arm_idx = state.agent.select_arm(x_agent)
        suggested_price = float(state.agent.prices[arm_idx])

    # 4. Generate transaction tracking ID
    request_id = str(uuid.uuid4())
    state.add_pending_request(request_id, arm_idx, x_agent, ctx)

    return PriceResponse(
        price=suggested_price,
        arm_id=arm_idx,
        request_id=request_id
    )


@app.post("/outcome", response_model=Acknowledgement)
async def report_outcome(payload: OutcomeInput):
    """
    Reports the purchase outcome (purchased: True/False) for a price offer.
    Triggers the online learning model parameter update.
    """
    request_id = payload.request_id

    # 1. Retrieve original context and arm index
    if request_id not in state.pending_requests:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction ID {request_id} not found or outcome already processed."
        )

    arm_idx, x_agent, ctx = state.pending_requests[request_id]
    chosen_price = state.agent.prices[arm_idx]

    # 2. Compute reward
    reward = chosen_price * (1.0 if payload.purchased else 0.0)

    # 3. Update agent parameters safely behind thread lock
    async with state.lock:
        state.agent.update(arm_idx, reward, x_agent)
        state.update_count += 1
        
        # Periodic snapshot saving
        if state.update_count % state.save_interval == 0:
            state.save_snapshot()

    # 4. Evict transaction mapping
    del state.pending_requests[request_id]

    return Acknowledgement(
        status="success",
        message=f"Outcome processed. Agent updated for arm {arm_idx} (price ${chosen_price:.2f}) with reward ${reward:.2f}."
    )
