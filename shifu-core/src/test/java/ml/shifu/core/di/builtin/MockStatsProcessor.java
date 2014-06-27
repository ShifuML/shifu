package ml.shifu.core.di.builtin;

import ml.shifu.core.container.RawValueObject;
import ml.shifu.core.container.obj.ColumnBinningResult;
import ml.shifu.core.container.obj.ColumnConfig;
import ml.shifu.core.di.spi.StatsProcessor;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class MockStatsProcessor implements StatsProcessor {

    public void process(ColumnConfig columnConfig, List<RawValueObject> rvoList) {
        ColumnBinningResult binningResult = new ColumnBinningResult();
        binningResult.setBinBoundary(Arrays.asList(Double.NEGATIVE_INFINITY, 1.0, 2.0, 10.0, 100.0));

        columnConfig.setColumnBinningResult(binningResult);
    }

    public void setParams(Map<String, Object> params) {

    }
}
