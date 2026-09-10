ALGORITHM: CausalSetMomentum

INPUT: 
    LOB stream {x_i} = (t_i, type_i, price_i, volume_i)
    Parameters: δ (causal window), θ (cluster threshold), 
                β (curvature sensitivity), γ (momentum decay)

OUTPUT:
    Directional signal s ∈ {-1, 0, +1}
    Target price level p_target

// Phase 1: Poset Construction
C ← ∅
for each event x_i:
    C ← C ∪ {x_i}
    for each y ∈ C:
        if causal_precedes(y, x_i):
            add_edge(y, x_i)  // y ≺ x_i
        end if
    end for
end for

// Phase 2: Local Geometry Computation
for each event x ∈ C:
    R(x) ← causal_curvature(x, window=δ)
    V_local(x) ← interval_volume(x, window=δ)
end for

// Phase 3: Horizon Detection
H ← ∅
for each event x:
    if ΔV_local(x) > threshold_horizon:
        H ← H ∪ {x}
    end if
end for

// Phase 4: Causal Attractor Computation
if H ≠ ∅:
    x_H ← argmax_{x ∈ H} |ΔV_local(x)|
    A ← {z : x_H ≺ z}
    Ω(p) ← Σ_{z ∈ A} interval_volume(x_H, z)
    p_attractor ← argmax_p Ω(p)
    
    // Phase 5: Momentum Signal
    s ← sign(p_attractor - p(x_H))
    p_target ← p_attractor
else:
    s ← 0
    p_target ← null
end if

// Phase 6: Cluster / Regime Shift Detection
K ← maximal_cluster(C, θ)
if cluster_fragmented(K, Δt):
    s ← -s  // regime shift reverses momentum
end if

// Phase 7: Bottleneck Filter
for each p ∈ price_range(p_target):
    if bottleneck_score(p) > threshold_bottleneck:
        p_target ← p
        break
    end if
end for

return (s, p_target)


