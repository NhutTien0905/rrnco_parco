# Workload Balancing Improvement Plan

## Problem Statement

The current HCVRP model exhibits **unbalanced workload distribution** among agents - some agents carry significantly more customers than others. This leads to:
- Suboptimal solutions (underutilized agents)
- Poor generalization to real-world scenarios requiring balanced workloads
- Potential capacity waste

## Root Causes Identified

1. **Reward Structure**: Only minimizes makespan, doesn't penalize imbalance
2. **Decoder Bias**: Softmax over all agents can create "greedy agent" phenomenon
3. **No Workload Tracking**: Environment doesn't track per-agent visit counts

---

## Solution 1: Add Workload Balancing to Reward

### Objective
Modify the reward function to penalize unbalanced workload distribution.

### Files to Modify
- `hybrid/hybrid/envs/hcvrp_real.py`

### Implementation Steps

1. **Track workload in `_step()` method**
   ```python
   # Calculate workload as: initial_capacity - remaining_capacity
   workload = td["capacity"] - td["capacity_remaining"]
   ```

2. **Modify `_get_reward()` method**
   ```python
   def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
       # Original: minimize makespan
       makespan = td["current_time"].max(dim=-1).values
       
       # Add: penalize workload imbalance (standard deviation)
       workload = td["capacity"] - td["capacity_remaining"]
       workload_std = workload.std(dim=-1)
       
       # Combined reward
       balance_weight = 0.1  # Hyperparameter to tune
       reward = -(makespan + balance_weight * workload_std)
       
       return reward
   ```

3. **Add hyperparameter to config**
   - File: `hybrid/configs/env/hcvrp_real.yaml`
   - Add: `balance_weight: 0.1`

### Pros & Cons
| Pros | Cons |
|------|------|
| Simple to implement | May conflict with makespan objective |
| End-to-end differentiable | Requires hyperparameter tuning |
| No architectural changes | May slow convergence initially |

### Testing
- Compare workload std before/after
- Monitor makespan to ensure it doesn't degrade significantly

---

## Solution 2: Modify Logits with Fairness Term

### Objective
Add a fairness penalty to the decoder logits to discourage over-utilized agents.

### Files to Modify
- `hybrid/hybrid/envs/hcvrp_real.py` (add visit tracking)
- `hybrid/hybrid/models/decoder.py` (add fairness penalty)

### Implementation Steps

1. **Add visit counter to environment state in `_reset()`**
   ```python
   td.update({
       # ...existing fields...
       "num_visited": torch.zeros(batch_size, self.num_agents, device=device),
   })
   ```

2. **Update visit counter in `_step()`**
   ```python
   # After selecting a customer (not depot)
   is_customer = selected >= self.num_agents  # Customer indices start at num_agents
   batch_idx = torch.arange(batch_size, device=device)
   agent_idx = td["current_agent"]
   
   # Increment visit count for active agent if visiting customer
   td["num_visited"][batch_idx, agent_idx] += is_customer.float()
   ```

3. **Add fairness penalty in `HybridDecoder.forward()`**
   ```python
   logits = self.pointer(glimpse, glimpse_key_static)  # [B, m, N]
   
   # Fairness penalty based on visit counts
   if "num_visited" in td:
       visit_counts = td["num_visited"]  # [B, m]
       mean_visits = visit_counts.mean(dim=-1, keepdim=True)  # [B, 1]
       fairness_penalty = (visit_counts - mean_visits).clamp(min=0)  # [B, m]
       fairness_penalty = fairness_penalty.unsqueeze(-1) * self.fairness_weight  # [B, m, 1]
       logits = logits - fairness_penalty
   
   logits = logits.masked_fill(~mask_combined, -torch.inf)
   ```

4. **Add fairness_weight as model parameter**
   - In `HybridDecoder.__init__()`: `self.fairness_weight = fairness_weight`
   - In config: `fairness_weight: 0.5`

### Pros & Cons
| Pros | Cons |
|------|------|
| Direct control during action selection | Adds complexity to decoder |
| Interpretable (penalize busy agents) | May need careful tuning |
| Can be toggled on/off easily | Slightly increases memory usage |

### Testing
- Visualize logits distribution with/without fairness
- Check if visit counts become more balanced

---

## Solution 3: Entropy Regularization for Agent Selection

### Objective
Add entropy regularization to encourage diverse agent selection.

### Files to Modify
- `hybrid/hybrid/models/policy.py`
- `hybrid/hybrid/models/decoder.py`

### Implementation Steps

1. **Calculate agent selection entropy in decoder**
   ```python
   # In HybridDecoder.forward()
   # After computing logits [B, m, N]
   
   # Marginalize over customers to get agent selection probabilities
   agent_logits = logits.logsumexp(dim=-1)  # [B, m]
   agent_probs = F.softmax(agent_logits, dim=-1)  # [B, m]
   
   # Calculate entropy: -sum(p * log(p))
   agent_entropy = -(agent_probs * agent_probs.log().clamp(min=-100)).sum(dim=-1)  # [B]
   
   # Store in td for loss calculation
   td["agent_entropy"] = agent_entropy
   ```

2. **Add entropy bonus to loss in policy**
   ```python
   # In REINFORCELoss or policy forward
   entropy_weight = 0.01  # Hyperparameter
   entropy_bonus = td["agent_entropy"].mean()
   
   # Total loss = policy_loss - entropy_weight * entropy_bonus
   # (negative because we want to maximize entropy)
   loss = policy_loss - entropy_weight * entropy_bonus
   ```

3. **Add config options**
   ```yaml
   # In model config
   entropy_weight: 0.01
   use_agent_entropy: true
   ```

### Pros & Cons
| Pros | Cons |
|------|------|
| Encourages exploration | May reduce solution quality initially |
| Prevents mode collapse | Indirect effect on balance |
| Standard RL technique | Entropy can decrease naturally over training |

### Testing
- Monitor agent_entropy during training
- Check if it leads to more balanced selection

---

## Implementation Priority

| Priority | Solution | Effort | Impact |
|----------|----------|--------|--------|
| 1 (High) | Solution 1: Reward Balancing | Low | High |
| 2 (Medium) | Solution 2: Fairness Logits | Medium | High |
| 3 (Low) | Solution 3: Entropy Regularization | Medium | Medium |

### Recommended Approach

1. **Start with Solution 1** - Simple reward modification
   - Quick to implement
   - Direct signal for balanced workload
   - Easy to evaluate impact

2. **Add Solution 2 if needed** - Fairness penalty
   - If reward alone isn't sufficient
   - Provides more direct control

3. **Consider Solution 3** - Entropy regularization
   - As supplementary technique
   - Helps with exploration

---

## Evaluation Metrics

### Primary Metrics
1. **Workload Standard Deviation**: `std(customers_per_agent)`
2. **Workload Coefficient of Variation**: `std / mean`
3. **Max/Min Ratio**: `max(workload) / min(workload)`

### Secondary Metrics (ensure no regression)
1. **Makespan**: Total completion time
2. **Total Distance**: Sum of all agent routes
3. **Feasibility Rate**: % of valid solutions

### Evaluation Script Location
- `hybrid/scripts/diagnose_workload.py` (create/update)

---

## Configuration Changes Summary

### `hybrid/configs/env/hcvrp_real.yaml`
```yaml
# Add workload balancing parameters
balance_weight: 0.1  # For Solution 1
```

### `hybrid/configs/model/hybrid.yaml`
```yaml
# Add fairness and entropy parameters
decoder:
  fairness_weight: 0.5  # For Solution 2
  
policy:
  entropy_weight: 0.01  # For Solution 3
  use_agent_entropy: true
```

---

## Testing Plan

### Unit Tests
- [ ] Test visit counter updates correctly
- [ ] Test fairness penalty computation
- [ ] Test entropy calculation

### Integration Tests
- [ ] Train for 10 epochs with each solution
- [ ] Compare workload distribution
- [ ] Verify no significant makespan regression

### Ablation Study
- [ ] Solution 1 only
- [ ] Solution 1 + 2
- [ ] Solution 1 + 2 + 3
- [ ] Vary hyperparameters (balance_weight, fairness_weight, entropy_weight)

---

## Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Implement Solution 1 | 1 day |
| 2 | Test & tune Solution 1 | 1 day |
| 3 | Implement Solution 2 | 1-2 days |
| 4 | Test & tune Solution 2 | 1 day |
| 5 | Implement Solution 3 (optional) | 1 day |
| 6 | Full evaluation & ablation | 2 days |

**Total Estimated Time**: 5-8 days
