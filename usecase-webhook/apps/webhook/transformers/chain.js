class SkipPublish extends Error {
  constructor(reason) {
    super(reason);
    this.name = 'SkipPublish';
  }
}

class TransformerChain {
  constructor() {
    this._transformers = [];
  }

  use(transformer) {
    if (typeof transformer.name !== 'string' || typeof transformer.transform !== 'function') {
      throw new Error('transformer must have { name: string, transform: function }');
    }
    this._transformers.push(transformer);
    return this;
  }

  // Runs all transformers in registration order.
  // context: { receivedAt, resolvedTopic, headers }
  // Returns the final payload, or throws SkipPublish to abort publishing.
  async run(payload, context) {
    let current = payload;
    for (const t of this._transformers) {
      current = await t.transform(current, context);
    }
    return current;
  }
}

module.exports = { TransformerChain, SkipPublish };
