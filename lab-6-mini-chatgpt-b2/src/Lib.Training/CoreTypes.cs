using System;

namespace Lib.Training
{
    public interface ILanguageModel
    {
        string ModelKind { get; }
    }

    public interface INGramModel : ILanguageModel
    {
        void Train(ReadOnlySpan<int> tokens);
    }

    public interface INeuralNetworkModel : ILanguageModel
    {
        double TrainStep(int[] context, int target, float learningRate);
    }
}