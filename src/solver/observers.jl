function _call_observer(observer, W, info::TimeStepInfo)
    if applicable(observer, W, info)
        return observer(W, info)
    elseif applicable(observer, W, info.k)
        return observer(W, info.k)
    else
        error("Observer must be callable as (W, info::TimeStepInfo) or (W, k::Int)")
    end
end

function compose_on_step(observers...)
    isempty(observers) && return nothing
    return (W, info) -> begin
        for observer in observers
            ret = _call_observer(observer, W, info)
            ret === false && return false
        end
        true
    end
end

