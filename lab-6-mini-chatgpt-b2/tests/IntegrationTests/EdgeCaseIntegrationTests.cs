using Lib.Batching;
using Lib.Batching.Streams;
using Lib.Training;
using Lib.Training.Configuration;

namespace IntegrationTests
{
    [TestFixture]
    public class EdgeCaseIntegrationTests
    {
        [Test] 
        public void TrainingLoop_WithZeroEpochs_DoesNotTrain()
        {
            var model = new MockTinyNN();
            var config = new TrainingConfig { Epochs = 0, StepsPerEpoch = 10 };
            var loop = new TrainingLoopImpl(model, new TokenBatchProvider(new ArrayTokenStream(new[] { 1, 2 })), config);

            loop.Run();

            Assert.That(model.TrainStepCalls, Is.EqualTo(0));
        }

        [Test] 
        public void TrainingLoop_WithBatchSizeOne_WorksInStochasticMode()
        {
            var model = new MockTinyNN();
            var config = new TrainingConfig { Epochs = 1, StepsPerEpoch = 3, BatchSize = 1, BlockSize = 1 };
            var loop = new TrainingLoopImpl(model, new TokenBatchProvider(new ArrayTokenStream(new[] { 1, 2, 3 })), config);

            loop.Run();

            Assert.That(model.TrainStepCalls, Is.EqualTo(3));
        }

        [Test]
        public void TrainingLoop_WithShortTokenStream_HandledGracefully()
        {
            var model = new MockTinyNN();
            var config = new TrainingConfig { Epochs = 1, StepsPerEpoch = 1, BatchSize = 1, BlockSize = 10 };
            var loop = new TrainingLoopImpl(model, new TokenBatchProvider(new ArrayTokenStream(new[] { 1, 2 })), config);

            Assert.DoesNotThrow(() => loop.Run());
        }
    }
}