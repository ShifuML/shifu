package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.container.obj.ColumnConfig;

import java.util.List;
import java.util.Map;

public interface StatsProcessor {

    public void process(ColumnConfig columnConfig, List<RawValueObject> rvoList);

    public void setParams(Map<String, Object> params);
}
