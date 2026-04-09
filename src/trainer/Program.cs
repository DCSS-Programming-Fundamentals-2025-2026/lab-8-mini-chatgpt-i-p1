using CommandLine;
using Lib.Batching;
using Lib.Corpus.Configuration;
using Lib.Corpus.Domain;
using Lib.Corpus.Infrastructure;
using Lib.Corpus.Processing;
using Lib.Tokenization.Interfaces;
using Lib.Tokenization.Model;
using Lib.Training;

namespace Trainer;

class Trainer
{
    public static void Main(string[] args)
    {
        string inputCommands = Console.ReadLine();
        Parser.Default.ParseArguments<Options>(args).WithParsed(RunOptions).WithNotParsed(HandleParseError);

        //Тут продовжує команда Б2
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
        }

        ITokenizer tokenizer = tokenizerFactory.BuildFromText(corpus.TrainText);
        int[] codedTrainTokens = tokenizer.Encode(corpus.TrainText);

        Console.WriteLine($"Токенізація успішна. Розмір словника: {tokenizer.VocabSize}");
        Console.WriteLine($"Всього отримано токенів: {codedTrainTokens.Length}");

        var batcher = new Batcher(tokens, opts.Seed);
        var batches = batcher.CreateBatches();

        ILanguageModel model = opts.Model switch
        {
            ModelType.Bigram => new BigramModel(),
            ModelType.Trigram => new TrigramModel(),
            ModelType.TinyNN => new TinyNNModel(opts.LearningRate), _ => throw new ArgumentException("Невідома модель")
        };

        for (int epoch = 1; epoch <= opts.Epochs; epoch++)
        {
            foreach (var batch in batches)
            {
                model.Train(batch);
            }

            CheckAndSave(epoch, model, opts.Out);

            Console.WriteLine($"Епоха {epoch}/{opts.Epochs} завершена.");
        }
    }

    public static void HandleParseError(IEnumerable<Error> errs)
    {
        Console.WriteLine("Incorrect arguments");
    }
}



