using NUnit.Framework;
using System;
using Lib.Batching;
using Lib.Batching.Streams;
using Lib.Training;
using Lib.Training.Configuration;

namespace IntegrationTests
{
    public class MockTinyNN : INeuralNetworkModel
    {
        public string ModelKind => "tinynn";
        public int TrainStepCalls { get; set; } = 0;
        public float LastLearningRate { get; set; }

        public double TrainStep(int[] context, int target, float learningRate)
        {
            TrainStepCalls++;
            LastLearningRate = learningRate;
            return 0.1;
        }
    }
    public class MockNGram : INGramModel
    {
        public string ModelKind => "ngram";
        public bool TrainWasCalled { get; set; } = false;

        public void Train(ReadOnlySpan<int> tokens)
        {
            TrainWasCalled = true;
        }
    }
}