struct BasicGate{SYM} end
for (SYM, OP) in [:∨ => (|), :∧ => (&), :⊻ => (⊻)]
    @eval nin(::BasicGate{$(QuoteNode(SYM))}) = 2
    @eval function (bg::BasicGate{$(QuoteNode(SYM))})(p, q)
        return $OP(p, q)
    end
end
nin(::BasicGate{:¬}) = 1
(bg::BasicGate{:¬})(p::T) where T = T(!Bool(p))

