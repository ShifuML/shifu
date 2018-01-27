package ml.shifu.shifu.udf;

import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.enhancer.EnhanceCallable;
import ml.shifu.shifu.enhancer.EnhancerFactory;
import ml.shifu.shifu.enhancer.EnhancerFactory.EnhanceType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import org.apache.pig.data.Tuple;

import java.io.IOException;
import java.util.*;

/**
 * EnhanceColumnUDF is used enhance column according to enhance index map and header
 * <p>
 * Author: Wu Devin (haifwu@paypal.com)
 * Date: 22/01/2018
 */
public class EnhanceColumnUDF extends AbstractTrainerUDF<String> {
    private Map<Integer, List<EnhanceType>> enhanceIndexMap = new HashMap<Integer, List<EnhanceType>>();
    private String[] headNames;
    private String dataDelimiter;
    private EnhancerFactory enhancerFactory = EnhancerFactory.getInstance();

    public EnhanceColumnUDF(String source, String pathModelConfig, String pathColumnConfig, String dataType, String
            evalSetName) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        if(! ("train".equals(dataType) || "eval".equals(dataType))){
            log.error("Not support enhance data type: " + dataType);
        }
        if(dataType.equals("train")){
            headNames = CommonUtils.getHeaders(modelConfig.getHeaderPath(), modelConfig.getHeaderDelimiter(),
                    modelConfig.getDataSet().getSource());
            dataDelimiter = modelConfig.getDataSetDelimiter();
        } else {
            RawSourceData evalData = modelConfig.getEvalConfigByName(evalSetName).getDataSet();
            headNames = CommonUtils.getHeaders(evalData.getHeaderPath(), evalData.getHeaderDelimiter(), evalData
                    .getSource());
            dataDelimiter = evalData.getDataDelimiter();
        }
        fullFillUniqueNoEmptyEnhanceMap(Constants.MAP_KEY_VALUE_DELIMITER, Constants.ENHANCE_METHOD_DELIMITER);
    }

    @Override
    public String exec(Tuple input) throws IOException {
        if(input == null) {
            return null;
        }
        String line = (String) input.get(0);
        if(line == null || line.isEmpty()){
            return line;
        }

        return enhance(line);
    }

    private String enhance(String line) {
        String[] dataColumns = line.split(this.dataDelimiter);
        // when data is not right the size of header, it's easy to enhance the wrong header
        if(headNames.length != dataColumns.length){
            return line;
        }

        List<String> dataList = Arrays.asList(dataColumns);
        for(Map.Entry<Integer, List<EnhanceType>> entry : enhanceIndexMap.entrySet()){
            int index = entry.getKey();
            Double originValue = null;
            try {
                originValue = Double.valueOf(dataList.get(index));
            } catch(NumberFormatException numberFormatException){
                log.warn("When enhance " + headNames[index] + " on " + dataList.get(index)
                        + ", get NumberFormatException!", numberFormatException);
            }
            for(EnhanceType enhancer : entry.getValue()){
                if(originValue != null){
                    dataList.add(String.valueOf(enhancerFactory.getEnhancer(enhancer).enhance(originValue)));
                } else {
                    dataList.add("");  // Add empty string for number convert failed field
                }
            }
        }
        return buildStringWithDelimiter(dataList, this.dataDelimiter);
    }

    private String buildStringWithDelimiter(List<String> strList, String delimiter){
        StringBuilder columns = new StringBuilder();
        for(String str: strList){
            columns.append(str).append(delimiter);
        }
        return columns.toString().substring(0, columns.length() - delimiter.length());
    }

    private String buildNewColumnName(String columnName, String enhancerName){
        return columnName + Constants.ENHANCE_NAME_JOIN_CHARACTER + enhancerName;
    }

    private Map<String, Integer> generateIndexMapFromHeaders(){
        Map<String, Integer> indexMap = new HashMap<String, Integer>();
        if(this.headNames == null) {
            return indexMap;
        }
        for(int i = 0; i < this.headNames.length; i++){
            indexMap.put(this.headNames[i].trim(), i);
        }
        return indexMap;
    }

    private void fullFillUniqueNoEmptyEnhanceMap(String configKeyValueDelimiter, String enhanceMethodDelimiter)
            throws IOException {
        Map<String, Integer> headerIndexMap = generateIndexMapFromHeaders();
        Map<String, String> configMap = CommonUtils.readConfFileIntoMap(this.modelConfig.getDataSet()
                .getEnhanceColumnFile(), this.modelConfig.getDataSet().getSource(), configKeyValueDelimiter);
        Set<String> enhanceTypeSet = new HashSet<String>();
        for(Map.Entry<String, String> entry : configMap.entrySet()) {
            if(entry.getValue() == null || entry.getValue().isEmpty() || entry.getKey() == null || entry.getKey().isEmpty()) {
                log.warn("Contain column name or enhanced method null or empty, ignored!");
                continue;
            }
            if(! headerIndexMap.containsKey(entry.getKey().trim())){
                log.warn("Column name " + entry.getKey() + " not in header, ignore enhance on this filed");
                continue;
            }

            // Use hashSet to generate unique enhance method for each column
            enhanceTypeSet.clear();
            if(entry.getValue().contains(enhanceMethodDelimiter)) {
                String[] methodList = enhanceMethodDelimiter.split(enhanceMethodDelimiter);
                for(String method : methodList) {
                    enhanceTypeSet.add(method.trim());
                }
            } else {
                enhanceTypeSet.add(entry.getValue().trim());
            }
            // remove empty string between two delimiter
            if(enhanceTypeSet.contains("")) {
                enhanceTypeSet.remove("");
            }

            List<EnhanceType> enhanceCallableList = new ArrayList<EnhanceType>();
            for(String enhanceType: enhanceTypeSet) {
                EnhanceType type = EnhanceType.fromString(enhanceType);
                if(type != null){
                    enhanceCallableList.add(type);
                }
            }

            // only put the key with none zero enhance method
            if(enhanceCallableList.size() > 0){
                this.enhanceIndexMap.put(headerIndexMap.get(entry.getKey()), enhanceCallableList);
            }
        }
    }

}
