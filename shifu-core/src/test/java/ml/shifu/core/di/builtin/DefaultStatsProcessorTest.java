package ml.shifu.core.di.builtin;


import ml.shifu.core.container.ColumnConfig;
import ml.shifu.core.di.spi.StatsProcessor;
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
