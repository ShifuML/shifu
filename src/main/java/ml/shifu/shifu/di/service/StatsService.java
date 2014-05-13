package ml.shifu.shifu.di.service;

import com.google.inject.Inject;
import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.di.spi.StatsProcessor;

import java.util.List;
import java.util.Map;

public class StatsService {

    private StatsProcessor statsProcessor;


    @Inject
    public StatsService(StatsProcessor processor) {
        this.statsProcessor = processor;
    }

    public void setParams(Map<String, Object> params) {
        this.statsProcessor.setParams(params);
    }

    public void exec(ColumnConfig columnConfig, List<RawValueObject> rvoList) {
        this.statsProcessor.process(columnConfig, rvoList);
    }
}
