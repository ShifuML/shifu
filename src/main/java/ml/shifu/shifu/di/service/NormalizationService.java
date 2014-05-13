package ml.shifu.shifu.di.service;

import com.google.inject.Inject;
import ml.shifu.shifu.di.spi.Normalizer;
import ml.shifu.shifu.container.obj.ColumnConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class NormalizationService {

    private static Logger log = LoggerFactory.getLogger(NormalizationService.class);
    private final Normalizer normalizer;

    @Inject
    public NormalizationService(Normalizer normalizer) {
        log.info("Normalizer Injected: " + normalizer.getClass().toString());
        this.normalizer = normalizer;
    }

    // Normalize a single object
    public Double normalize(ColumnConfig config, Object raw) {
        return normalizer.normalize(config, raw);
    }

    // Normalize a list of objects
    public List<Double> normalize(List<ColumnConfig> columns, List<Object> values) {

        if (columns.size() != values.size()) {
            throw new IllegalArgumentException("Columns size should equal values size: columns size = " + columns.size() + ", values size = " + values.size());
        }

        List<Double> result = new ArrayList<Double>();

        Integer size = columns.size();
        for (int i = 0; i < size; i++) {
            if (columns.get(i).isFinalSelect()) {
                if (values.get(i) == null) {
                    return null;
                }

                result.add(normalize(columns.get(i), values.get(i)));
            }
        }

        return result;
    }

    // Normalize a rawDataMap
    public List<Double> normalize(List<ColumnConfig> columnConfigList, Map<String, ? extends Object> rawDataMap) {
        List<Double> result = new ArrayList<Double>();
        for (ColumnConfig config : columnConfigList) {
            String key = config.getColumnName();
            if ((config.isFinalSelect() || config.isTarget()) && !rawDataMap.containsKey(key)) {
                throw new IllegalStateException(String.format("Variable Missing: %s", key));
            }

            if (config.isFinalSelect()) {
                result.add(normalize(config, rawDataMap.get(key).toString()));
            }
        }
        return result;
    }

}
