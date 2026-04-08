using System;
using Lib.Batching; 
using Lib.Training.Configuration;
using Lib.Training.Metrics;
using Lib.Training.Scheduling;

namespace Lib.Training
{
    public class TrainingLoopImpl : ITrainingLoop
    {
        private readonly ILanguageModel _model;
        private readonly IBatchProvider _batchProvider;
        private readonly TrainingConfig _config;
        private readonly TrainingMetrics? _metrics;
        private readonly CheckpointScheduler? _scheduler;
        private readonly Random _rng;

        public TrainingLoopImpl(
            ILanguageModel model,
            IBatchProvider batchProvider,
            TrainingConfig config,
            TrainingMetrics? metrics = null,
            CheckpointScheduler? scheduler = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _batchProvider = batchProvider ?? throw new ArgumentNullException(nameof(batchProvider));
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _metrics = metrics;
            _scheduler = scheduler;
            _rng = new Random();
        }

        public void Run()
        {
            if (_config.Epochs <= 0) return;

            if (_model is INeuralNetworkModel nnModel)
            {
                for (int epoch = 0; epoch < _config.Epochs; epoch++)
                {
                    double epochLoss = 0;
                    int batchesProcessed = 0;

                    while (batchesProcessed < _config.StepsPerEpoch)
                    {
                        var batch = _batchProvider.GetBatch(_config.BatchSize, _config.BlockSize, _rng);
                        if (batch == null || batch.Contexts.Length == 0) break;

                        double batchLoss = 0;

                        for (int i = 0; i < batch.Contexts.Length; i++)
                        {
                            batchLoss += nnModel.TrainStep(batch.Contexts[i], batch.Targets[i], _config.LearningRate);
                        }

                        epochLoss += (batchLoss / batch.Contexts.Length);
                        batchesProcessed++;
                    }

                    double averageLoss = batchesProcessed > 0 ? epochLoss / batchesProcessed : 0;
                    _metrics?.RecordEpoch(epoch, averageLoss);
                    _scheduler?.CheckAndSave(epoch, _model);
                }
            }
            else if (_model is INGramModel ngramModel)
            {
                try
                {
                    if (_batchProvider is TokenBatchProvider provider)
                    {
                        var allTokens = provider.Stream.GetTokens();
                        ngramModel.Train(allTokens);

                        _metrics?.RecordEpoch(0, 0.0); 
                        _scheduler?.CheckAndSave(0, _model);

                        Console.WriteLine($"[B2] Модель {ngramModel.ModelKind} успішно навчена на {allTokens.Length} токенах.");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[B2 Error] Помилка під час тренування NGram: {ex.Message}");
                    throw; 
                }
            }
        }
    }
}