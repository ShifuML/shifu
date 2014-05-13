package ml.shifu.shifu.di.builtin;


import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.di.module.NormalizationModule;
import ml.shifu.shifu.di.service.NormalizationService;
import ml.shifu.shifu.di.spi.StatsProcessor;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

import java.util.Arrays;

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
