package ml.shifu.shifu.combo;

import ml.shifu.shifu.container.obj.ModelBasicConf;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Created by zhanhu on 12/9/16.
 */
public class DataMerger {

    private static Logger LOG = LoggerFactory.getLogger(DataMerger.class);

    private ModelBasicConf.RunMode runMode;
    private String joinColumnName;
    private String outputDataPath;

    /**
     * ColumnFile list to merge
     */
    private List<ColumnFile> columnFileList;

    public DataMerger(ModelBasicConf.RunMode runMode, String joinColumnName, String outputDataPath) {
        this.runMode = runMode;
        this.joinColumnName = joinColumnName;
        this.outputDataPath = outputDataPath;
        this.columnFileList = new ArrayList<ColumnFile>();
    }

    public void addColumnFile(ColumnFile columnFile) {
        this.columnFileList.add(columnFile);
    }

    public void addColumnFileList(List<ColumnFile> columnFile) {
        this.columnFileList.addAll(columnFile);
    }

    public boolean doMerge() throws IOException {
        if ( ModelBasicConf.RunMode.LOCAL.equals(runMode) ) {
            // do local data merge
            genOutputHeader();
            mergeData();
        } else if ( ModelBasicConf.RunMode.MAPRED.equals(runMode)
                || ModelBasicConf.RunMode.DIST.equals(runMode) ) {
            // use pig to do data merge
            runMapReduceToMerge();
        } else {
            throw new ShifuException(ShifuErrorCode.ERROR_UNSUPPORT_MODE);
        }

        return true;
    }

    private void runMapReduceToMerge() throws IOException {
        PigDataJoin inst = new PigDataJoin();
        inst.join(joinColumnName, outputDataPath, columnFileList);
    }

    private void genOutputHeader() throws IOException {
        List<String> outputHeaders = new ArrayList<String>();
        for ( ColumnFile columnFile : columnFileList ) {
            outputHeaders.addAll(columnFile.getOutputVarNames());
        }

        String header = StringUtils.join(outputHeaders, "|");
        File outputDir = new File(this.outputDataPath);
        if(!outputDir.mkdirs()){
            LOG.error("Create folder {} failed.", outputDir);
        }

        FileUtils.write(new File(this.outputDataPath + File.separator + ".pig_header"), header);
    }

    private void mergeData() throws IOException {
        Map<String, List<String>> data = this.columnFileList.get(0).loadSelectedData(joinColumnName);
        LOG.info("load {} records from {}.", data.size(), this.columnFileList.get(0).getFilePath());

        List<Map<String, List<String>>> selectedDatas = new ArrayList<Map<String, List<String>>>();
        for ( int i = 1; i < this.columnFileList.size(); i ++ ) {
            Map<String, List<String>> selectedData = this.columnFileList.get(i).loadSelectedData(joinColumnName);
            selectedDatas.add(selectedData);
            LOG.info("load {} records from {}.", selectedData.size(), this.columnFileList.get(i).getFilePath());
        }

        Iterator<Map.Entry<String, List<String>>> iterator = data.entrySet().iterator();
        while ( iterator.hasNext() ) {
            Map.Entry<String, List<String>> entry = iterator.next();
            String key = entry.getKey();
            List<String> vars = entry.getValue();

            for (Map<String, List<String>> selectedData : selectedDatas) {
                if ( selectedData.containsKey(key) ) {
                    vars.addAll(selectedData.get(key));
                } else {
                    iterator.remove();
                    break;
                }
            }
        }

        File dataPart = new File(this.outputDataPath + File.separator + "part-00");
        List<String> records = new ArrayList<String>();
        for (List<String> varList : data.values()) {
            records.add(StringUtils.join(varList, "|"));
        }
        FileUtils.writeLines(dataPart, records);
    }
}
