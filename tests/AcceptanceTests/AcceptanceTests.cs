using MiniChatGPT.Contracts;

namespace AcceptanceTests;

public class Tests
{
    private string _testDataPath;
    private string _trigramOutPath;
    private string _tinynnOutPath;

    [SetUp]
    public void Setup()
    {
        _testDataPath = Path.GetTempFileName();
        File.WriteAllText(_testDataPath, "це текстовий корпус для тестування Mini ChatGPT та іншого тренера");
        _trigramOutPath = Path.Combine(Path.GetTempPath(), "test_trigram.json");
        _tinynnOutPath = Path.Combine(Path.GetTempPath(), "test_tinynn.json");
    }

    [TearDown]
    public void TearDown()
    {
        if (File.Exists(_testDataPath)) File.Delete(_testDataPath);
        if (File.Exists(_trigramOutPath)) File.Delete(_trigramOutPath);
        if (File.Exists(_tinynnOutPath)) File.Delete(_tinynnOutPath);
    }

    [Test]
    public void Trainer_TrainTrigram_CheckpointSavedAndContainsCorrectFields()
    {
        var args = new[] { "--data", _testDataPath, "--model", "trigram", "--tokenizer", "word", "--out", _trigramOutPath, "--seed", "42" };
        Trainer.Trainer.Main(args);

        Assert.That(File.Exists(_trigramOutPath), Is.True, "Checkpoint file for trigram should exist.");

        var checkpointIo = new JsonCheckpointIO();
        var checkpoint = checkpointIo.Load(_trigramOutPath);

        Assert.That(checkpoint.ModelKind, Is.EqualTo("trigram"));
        Assert.That(checkpoint.TokenizerKind, Is.EqualTo("word"));
        Assert.That(checkpoint.Seed, Is.EqualTo(42));
    }

    [Test]
    public void Trainer_TrainTinyNN_CheckpointSavedAndContainsCorrectFields()
    {
        var args = new[] { "--data", _testDataPath, "--model", "tinynn", "--tokenizer", "word", "--out", _tinynnOutPath, "--seed", "123", "--epochs", "1" };
        Trainer.Trainer.Main(args);

        Assert.That(File.Exists(_tinynnOutPath), Is.True, "Checkpoint file for tinynn should exist.");

        var checkpointIo = new JsonCheckpointIO();
        var checkpoint = checkpointIo.Load(_tinynnOutPath);

        Assert.That(checkpoint.ModelKind, Is.EqualTo("tinynn"));
        Assert.That(checkpoint.TokenizerKind, Is.EqualTo("word"));
        Assert.That(checkpoint.Seed, Is.EqualTo(123));
    }
}
