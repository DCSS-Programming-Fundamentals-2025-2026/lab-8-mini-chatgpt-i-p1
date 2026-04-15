using CommandLine;
using Lib.Batching;
using Lib.Batching.Streams;
using Lib.Corpus.Configuration;
using Lib.Corpus.Domain;
using Lib.Corpus.Infrastructure;
using Lib.Corpus.Processing;
using Lib.MathCore;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration; 
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyTransformer;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Factories;
using Lib.Tokenization.Interfaces;
using Lib.Tokenization.Model;
using MiniChatGPT.Contracts;
using NGram;
using NGram.ModelFactory;

namespace Trainer;

public class Trainer
{
    public static void Main(string[] args)
    {
        Parser.Default.ParseArguments<Options>(args).WithParsed(RunOptions).WithNotParsed(HandleParseError);
    }

    public static void RunOptions(Options opts)
    {
        Console.WriteLine($"Your model: {opts.Model}");

        string TokenizerVer = "Unknown";
        string CorpusVer = "Unknown";
        string ModelVer = "Unknown";

        var splitter = new CorpusSplitter();
        var normilizer = new CorpusTextNormalizer();
        var fileSystem = new DefaultFileSystem();
        var loader = new CorpusLoader(normilizer, splitter, fileSystem);

        CorpusVer = loader.GetContractFingerprint();

        string dataPath = opts.Data;

        if (!File.Exists(dataPath))
        {
            Console.WriteLine("Файл не знайдено");
            return;
        }
        
        

        CorpusLoadOptions loadOptions = new();
        var corpus = loader.Load(dataPath, loadOptions);

        Console.WriteLine($"Корпус довжиною {corpus?.TrainText.Length} Завантажено");
        ITokenizerFactory tokenizerFactory = new WordTokenizerFactory();

        if (opts.Tokenizer == "word")
        {
            tokenizerFactory = new WordTokenizerFactory();
        }
        else if (opts.Tokenizer == "char")
        {
            tokenizerFactory = new CharTokenizerFactory();
        }
        else
        {
            Console.WriteLine("Invalid tokenizer type.");
            return;
        }

        Lib.Tokenization.Interfaces.ITokenizer tokenizer = tokenizerFactory.BuildFromText(corpus.TrainText);
        TokenizerVer = tokenizer.GetContractFingerprint();
        int[] codedTrainTokens = tokenizer.Encode(corpus.TrainText);

        Console.WriteLine($"Токенізація успішна. Розмір словника: {tokenizer.VocabSize}");
        Console.WriteLine($"Всього отримано токенів: {codedTrainTokens.Length}");


        ArrayTokenStream tokenStream = new ArrayTokenStream(codedTrainTokens);
        TokenBatchProvider batchProvider = new TokenBatchProvider(tokenStream);
        NGramModelFactory modelFactory = new NGramModelFactory();
        JsonCheckpointIO json = new JsonCheckpointIO();
        Random rng = new(opts.Seed);
        Batch batches = batchProvider.GetBatch(32, 8, rng);

        IMathOps mathOps = new MathOpsImpl();

        if (opts.Model.ToLower() == "tinynn")
        {
            var nnFactory = new TinyNNModelFactory();
            var nnConfig = new TinyNNConfig(tokenizer.VocabSize, 64);
            TinyNNModel model = nnFactory.CreateNew(nnConfig, mathOps);
            ModelVer = model.GetContractFingerprint();
            Console.WriteLine("TinyNN created successfully!");

            for (int i = 0; i < opts.Epochs; i++)
            {
                float totalLoss = 0;
                int contextSize = 8; 
                for (int j = 0; j < codedTrainTokens.Length - contextSize; j++)
                {
                    ReadOnlySpan<int> context = new ReadOnlySpan<int>(codedTrainTokens, j, contextSize);
                    int target = codedTrainTokens[i + contextSize];
                    float loss = model.TrainStep(context, target, opts.LearningRate);
                    totalLoss += loss;
                }
            }

            Checkpoint checkpoint = new Checkpoint(opts.Model, opts.Tokenizer, tokenizer.GetPayloadForCheckpoint(), model.ToPayload(), opts.Seed, GenerateFingerprintChain(CorpusVer, TokenizerVer, ModelVer));
            json.Save(opts.Out, checkpoint);
        }
        else if (opts.Model.ToLower() == "tinytransformer")
        {
            var tfConfig = new TinyTransformerConfig(tokenizer.VocabSize, 64, 2, 8, opts.Seed);
            TinyTransformerModel model = TinyTransformerModelFactory.CreateAuto(tfConfig);
            ModelVer = model.GetContractFingerprint();
            Console.WriteLine("TinyTransformer created successfully!");

            Checkpoint checkpoint = new Checkpoint(opts.Model, opts.Tokenizer, tokenizer.GetPayloadForCheckpoint(), model.GetPayloadForCheckpoint(), opts.Seed, GenerateFingerprintChain(CorpusVer, TokenizerVer, ModelVer));
            json.Save(opts.Out, checkpoint);

        }
        else if (opts.Model.ToLower() == "trigram")
        {
            TrigramModel trigram = modelFactory.CreateTrigramModel((tokenizer.VocabSize));
            ModelVer = trigram.GetContractFingerprint();
            Console.WriteLine("Trigram created successfully!");

            trigram.Train(codedTrainTokens);

            Checkpoint checkpoint = new Checkpoint(opts.Model, opts.Tokenizer, tokenizer.GetPayloadForCheckpoint(), trigram.GetPayloadForCheckpoint(), opts.Seed, GenerateFingerprintChain(CorpusVer, TokenizerVer, ModelVer));
            json.Save(opts.Out, checkpoint);

        }
        else if (opts.Model.ToLower() == "bigram")
        {
            NGramModel bigram = modelFactory.CreateBigramModel((tokenizer.VocabSize));
            ModelVer = bigram.GetContractFingerprint();
            Console.WriteLine("Bigram created successfully!");

            bigram.Train(codedTrainTokens);

            Checkpoint checkpoint = new Checkpoint(opts.Model, opts.Tokenizer, tokenizer.GetPayloadForCheckpoint(), bigram.GetPayloadForCheckpoint(), opts.Seed, GenerateFingerprintChain(CorpusVer, TokenizerVer, ModelVer));
            json.Save(opts.Out, checkpoint);
        }
        else
        {
            Console.WriteLine("Incorrect model");
        }



    }

    public static void HandleParseError(IEnumerable<Error> errs)
    {
        Console.WriteLine("Incorrect arguments");
    }

    private static string GenerateFingerprintChain(string corpusLoaderVer, string tokenizerVer, string modelVer)
    {
        return $"Lib.Corpus: {corpusLoaderVer}|Lib.Tokenization:{tokenizerVer}|Lib.Model:{modelVer}";
    }
}