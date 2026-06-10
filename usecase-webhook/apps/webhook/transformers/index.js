const { TransformerChain } = require('./chain');

// ── Transformer examples (commented out) ─────────────────────────────────────
//
// Each transformer receives (payload, context) and returns the (modified) payload.
// Throw SkipPublish to abort publishing without treating it as an error.
//
// const { SkipPublish } = require('./chain');
//
// const addReceivedAt = {
//   name: 'addReceivedAt',
//   transform(payload, context) {
//     return { ...payload, _received_at: context.receivedAt };
//   },
// };
//
// const rejectIfMissingId = {
//   name: 'rejectIfMissingId',
//   transform(payload, context) {
//     if (!payload.id) throw new SkipPublish('missing required field: id');
//     return payload;
//   },
// };
//
// ─────────────────────────────────────────────────────────────────────────────

const chain = new TransformerChain();

// Register transformers here in order:
// chain.use(addReceivedAt);
// chain.use(rejectIfMissingId);

module.exports = chain;
