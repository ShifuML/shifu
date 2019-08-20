package ml.shifu.shifu.core.dtrain.mtl;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.SerializationType;
import ml.shifu.shifu.core.dtrain.StringUtils;
import ml.shifu.shifu.core.dtrain.nn.NNColumnStats;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

/**
 * @author haillu
 */
public class BinaryMTLSerializer {
    public static void save(ModelConfig modelConfig, List<List<ColumnConfig>> mtlColumnConfigLists,
            MultiTaskLearning mtl, FileSystem fs, Path output) throws IOException {
        DataOutputStream fos = null;
        try {
            fos = new DataOutputStream(new GZIPOutputStream(fs.create(output)));

            // version
            fos.writeInt(CommonConstants.MTL_FORMAT_VERSION);
            // Reserved two double field, one double field and one string field
            fos.writeDouble(0.0f);
            fos.writeDouble(0.0f);
            fos.writeDouble(0.0d);
            fos.writeUTF("Reserved field");

            // write normStr
            String normStr = modelConfig.getNormalize().getNormType().toString();
            StringUtils.writeString(fos, normStr);

            // write task number.
            fos.writeInt(mtlColumnConfigLists.size());

            for(List<ColumnConfig> ccs: mtlColumnConfigLists) {
                // compute columns needed
                Map<Integer, String> columnIndexNameMapping = getIndexNameMapping(ccs);

                // write column stats to output
                List<NNColumnStats> csList = new ArrayList<>();
                for(ColumnConfig cc: ccs) {
                    if(columnIndexNameMapping.containsKey(cc.getColumnNum())) {
                        NNColumnStats cs = new NNColumnStats();
                        cs.setCutoff(modelConfig.getNormalizeStdDevCutOff());
                        cs.setColumnType(cc.getColumnType());
                        cs.setMean(cc.getMean());
                        cs.setStddev(cc.getStdDev());
                        cs.setColumnNum(cc.getColumnNum());
                        cs.setColumnName(cc.getColumnName());
                        cs.setBinCategories(cc.getBinCategory());
                        cs.setBinBoundaries(cc.getBinBoundary());
                        cs.setBinPosRates(cc.getBinPosRate());
                        cs.setBinCountWoes(cc.getBinCountWoe());
                        cs.setBinWeightWoes(cc.getBinWeightedWoe());

                        // TODO cache such computation
                        double[] meanAndStdDev = Normalizer.calculateWoeMeanAndStdDev(cc, false);
                        cs.setWoeMean(meanAndStdDev[0]);
                        cs.setWoeStddev(meanAndStdDev[1]);
                        double[] weightMeanAndStdDev = Normalizer.calculateWoeMeanAndStdDev(cc, true);
                        cs.setWoeWgtMean(weightMeanAndStdDev[0]);
                        cs.setWoeWgtStddev(weightMeanAndStdDev[1]);

                        csList.add(cs);
                    }
                }

                fos.writeInt(csList.size());
                for(NNColumnStats cs: csList) {
                    cs.write(fos);
                }

                Map<Integer, Integer> columnMapping = DTrainUtils.getColumnMapping(ccs);
                fos.writeInt(columnMapping.size());
                for(Map.Entry<Integer, Integer> entry: columnMapping.entrySet()) {
                    fos.writeInt(entry.getKey());
                    fos.writeInt(entry.getValue());
                }

            }

            // persist multi task learning Model
            mtl.write(fos, SerializationType.MODEL_SPEC);
        } finally {
            IOUtils.closeStream(fos);
        }
    }

    private static Map<Integer, String> getIndexNameMapping(List<ColumnConfig> columnConfigList) {
        Map<Integer, String> columnIndexNameMapping = new HashMap<>(columnConfigList.size());
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.isFinalSelect()) {
                columnIndexNameMapping.put(columnConfig.getColumnNum(), columnConfig.getColumnName());
            }
        }

        if(columnIndexNameMapping.size() == 0) {
            boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
            for(ColumnConfig columnConfig: columnConfigList) {
                if(CommonUtils.isGoodCandidate(columnConfig, hasCandidates)) {
                    columnIndexNameMapping.put(columnConfig.getColumnNum(), columnConfig.getColumnName());
                }
            }
        }
        return columnIndexNameMapping;
    }
}
