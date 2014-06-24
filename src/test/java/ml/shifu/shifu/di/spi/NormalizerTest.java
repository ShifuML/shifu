package ml.shifu.shifu.di.spi;


import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.di.builtin.WOENormalizer;
import ml.shifu.shifu.di.builtin.ZScoreNormalizer;
import ml.shifu.shifu.di.module.NormalizationModule;
import ml.shifu.shifu.di.service.NormalizationService;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

import java.util.Arrays;

public class NormalizerTest {

    private ColumnConfig config;

    @BeforeTest
    public void setUp() {

        config = new ColumnConfig();

        config.getColumnBinningResult().setBinBoundary(Arrays.asList(Double.NEGATIVE_INFINITY, 1.0, 5.0, 10.0));
        config.getColumnBinStatsResult().setBinWoe(Arrays.asList(-1.0, 2.0, -3.0, 4.0));
        config.setMean(10.0);
        config.setStdDev(2.0);

    }

    @Test
    public void testWOENormalizer() {

        NormalizationService normalizationService = new NormalizationService(new WOENormalizer());
        Assert.assertEquals(normalizationService.normalize(config, 2.5), 2.0);
    }

    @Test
    public void testZScoreNormalizer() {


        NormalizationService normalizationService = new NormalizationService(new ZScoreNormalizer());
        Assert.assertEquals(normalizationService.normalize(config, 2.5), -3.75);
    }

    @Test
    public void testInjector() {
        // WOE
        Injector injector = Guice.createInjector(new NormalizationModule(WOENormalizer.class.getName()));
        NormalizationService normalizationService = injector.getInstance(NormalizationService.class);
        Assert.assertEquals(normalizationService.normalize(config, 2.5), 2.0);

        // ZScore
        injector = Guice.createInjector(new NormalizationModule(ZScoreNormalizer.class.getName()));
        normalizationService = injector.getInstance(NormalizationService.class);
        Assert.assertEquals(normalizationService.normalize(config, 2.5), -3.75);
    }

    public void testNormalizationWorker() {

    }
}
