using CommandLine;
using Lib.Batching;
using Lib.Batching.Streams;
using Lib.Corpus.Configuration;
using Lib.Corpus.Domain;
using Lib.Corpus.Infrastructure;
using Lib.Corpus.Processing;
using Lib.Tokenization.Interfaces;
using Lib.Tokenization.Model;
using Lib.Training;

using Lib.MathCore;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyNN.Configuration; 
using Lib.Models.TinyTransformer.Factories;
using Lib.Models.TinyTransformer.Configuration;

namespace Trainer;

class Trainer
{
    public static void Main(string[] args)
    {
        Parser.Default.ParseArguments<Options>(args).WithParsed(RunOptions).WithNotParsed(HandleParseError);
    }

    public static void RunOptions(Options opts)
    {
        Console.WriteLine($"Your model: {opts.Model}");

        var splitter = new CorpusSplitter();
        var normilizer = new CorpusTextNormalizer();
        var fileSystem = new DefaultFileSystem();
        var loader = new CorpusLoader(normilizer, splitter, fileSystem);

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

        ITokenizer tokenizer = tokenizerFactory.BuildFromText(corpus.TrainText);
        int[] codedTrainTokens = tokenizer.Encode(corpus.TrainText);

        Console.WriteLine($"Токенізація успішна. Розмір словника: {tokenizer.VocabSize}");
        Console.WriteLine($"Всього отримано токенів: {codedTrainTokens.Length}");


        ArrayTokenStream tokenStream = new ArrayTokenStream(codedTrainTokens);
        TokenBatchProvider batchProvider = new TokenBatchProvider(tokenStream);
        Random rng = new(opts.Seed);
        Batch batches = batchProvider.GetBatch(32, 8, rng);

        IMathOps mathOps = new MathOpsImpl();
        object model = null; 

        if (opts.Model.ToLower() == "tinynn")
        {
            var nnFactory = new TinyNNModelFactory();
            var nnConfig = new TinyNNConfig(tokenizer.VocabSize, 64);
            model = nnFactory.CreateNew(nnConfig, mathOps);
            Console.WriteLine("TinyNN created successfully!");
        }
        else if (opts.Model.ToLower() == "tinytransformer")
        {
            var tfConfig = new TinyTransformerConfig(tokenizer.VocabSize, 64, 2, 8, opts.Seed);
            model = TinyTransformerModelFactory.CreateAuto(tfConfig);
            Console.WriteLine("TinyTransformer created successfully!");
        }
    }

    public static void HandleParseError(IEnumerable<Error> errs)
    {
        Console.WriteLine("Incorrect arguments");
    }
}