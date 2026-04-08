using Lib.Batching;
using Lib.Batching.Streams;
using Lib.Training;
using Lib.Training.Configuration;

namespace IntegrationTests
{
    [TestFixture]
    public class StabilityIntegrationTests
    {
        [Test] 
        public void SeededRng_ProducesIdenticalBatches()
        {
            var tokens = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var stream = new ArrayTokenStream(tokens);
            var provider = new TokenBatchProvider(stream);

            var rng1 = new Random(42);
            var rng2 = new Random(42);

            var batch1 = provider.GetBatch(2, 2, rng1);
            var batch2 = provider.GetBatch(2, 2, rng2);

            Assert.That(batch1.Targets, Is.EqualTo(batch2.Targets), "Батчі мають бути ідентичними при однаковому Seed");
        }

        [Test] 
        public void TrainingLoop_WithEmptyStream_ThrowsException()
        {
            var provider = new TokenBatchProvider(new ArrayTokenStream(Array.Empty<int>()));
            var loop = new TrainingLoopImpl(new MockTinyNN(), provider, new TrainingConfig());

            Assert.Throws<InvalidOperationException>(() => loop.Run());
        }

        [Test] 
        public void TrainingLoop_Initialization_RequiresAllComponents()
        {
            Assert.Throws<ArgumentNullException>(() => new TrainingLoopImpl(null!, null!, null!));
        }
    }
}