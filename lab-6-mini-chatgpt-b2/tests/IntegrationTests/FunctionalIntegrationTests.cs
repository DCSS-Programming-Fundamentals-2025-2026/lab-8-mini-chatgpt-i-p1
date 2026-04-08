using Lib.Batching;
using Lib.Batching.Streams;
using Lib.Training;
using Lib.Training.Configuration;

namespace IntegrationTests
{
    [TestFixture]
    public class FunctionalIntegrationTests
    {
        [Test] 
        public void TinyNN_TrainingCycle_ExecutesCorrectNumberOfSteps()
        {
            var stream = new ArrayTokenStream(new[] { 1, 2, 3, 4, 5, 6 });
            var provider = new TokenBatchProvider(stream);
            var model = new MockTinyNN();
            var config = new TrainingConfig { Epochs = 2, StepsPerEpoch = 2, BatchSize = 2, BlockSize = 2 };

            var loop = new TrainingLoopImpl(model, provider, config);
            loop.Run();

            Assert.That(model.TrainStepCalls, Is.EqualTo(8));
        }

        [Test] 
        public void NGram_Training_CallsSpecificTrainMethod()
        {
            var stream = new ArrayTokenStream(new[] { 1, 2, 3 });
            var provider = new TokenBatchProvider(stream);
            var model = new MockNGram();
            var config = new TrainingConfig { Epochs = 1 };

            var loop = new TrainingLoopImpl(model, provider, config);
            loop.Run();

            Assert.That(model.TrainWasCalled, Is.True, "Метод Train для NGram не був викликаний");
        }

        [Test] 
        public void TrainingConfig_Parameters_ArePassedToModel()
        {
            var provider = new TokenBatchProvider(new ArrayTokenStream(new[] { 1, 2, 3, 4 }));
            var model = new MockTinyNN();
            float expectedLr = 0.05f;
            var config = new TrainingConfig { Epochs = 1, StepsPerEpoch = 1, BatchSize = 1, LearningRate = expectedLr };

            new TrainingLoopImpl(model, provider, config).Run();

            Assert.That(model.LastLearningRate, Is.EqualTo(expectedLr));
        }
    }
}