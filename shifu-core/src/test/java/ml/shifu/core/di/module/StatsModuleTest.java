package ml.shifu.core.di.module;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.container.RawValueObject;
import ml.shifu.core.container.obj.ColumnConfig;
import ml.shifu.core.di.service.StatsService;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.*;

public class StatsModuleTest {

    @Test
    public void testMockStatsProcessor() {

        Map<String, String> methods = new HashMap<String, String>();
        methods.put("ColumnRawStatsCalculator", "ml.core.core.di.builtin.DefaultColumnRawStatsCalculator");
        methods.put("StatsProcessor", "ml.core.core.di.builtin.MockStatsProcessor");

        StatsModule statsModule = new StatsModule();
        statsModule.setInjections(methods);

        Injector injector = Guice.createInjector(statsModule);
        StatsService statsService = injector.getInstance(StatsService.class);

        ColumnConfig columnConfig = new ColumnConfig();
        columnConfig.setColumnType(ColumnConfig.ColumnType.N);
        statsService.exec(columnConfig, genData1());

        Assert.assertEquals(columnConfig.getColumnBinningResult().getBinBoundary(), Arrays.asList(Double.NEGATIVE_INFINITY, 1.0, 2.0, 10.0, 100.0));

    }

    public void testMock() {
        /*statsModule.setNumBinningCalculatorImplClass(TotalPercentileColumnNumBinningCalculator.class);
        statsModule.setCatBinningCalculatorImplClass(SimpleColumnCatBinningCalculator.class);
        statsModule.setNumStatsCalculatorImplClass(DefaultColumnNumStatsCalculator.class);
        statsModule.setBinStatsCalculatorImplClass(DefaultColumnBinStatsCalculator.class);
        statsModule.setRawStatsCalculatorImplClass(MockColumnRawStatsCalculator.class);
         */


        Map<String, Object> params = new HashMap<String, Object>();
      /*
        params.put("posTags", Arrays.asList("P"));
        params.put("negTags", Arrays.asList("N"));
        params.put("numBins", 4);

        */


        //statsService.setGlobalParams(params);


        //System.out.println(config.getBinBoundary());
    }

    private List<RawValueObject> genData1() {
        List<RawValueObject> rvoList = new ArrayList<RawValueObject>();

        for (int i = 0; i < 10; i++) {
            RawValueObject rvo = new RawValueObject();
            rvo.setValue(i);
            rvo.setTag("P");
            rvo.setWeight(1.0);
            rvoList.add(rvo);
        }
        return rvoList;
    }

}
