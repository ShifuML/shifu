package ml.shifu.shifu.di.builtin;


import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.di.spi.StatsProcessor;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class DefaultStatsProcessorTest {

    private ColumnConfig config;

    @BeforeTest
    public void setUp() {
        StatsProcessor processor = new DefaultStatsProcessor(null, null, null, null, null);

    }

    @Test
    public void testSetParams() {

    }


    public void testNormalizationWorker() {

    }
}
