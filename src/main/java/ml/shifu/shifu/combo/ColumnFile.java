package ml.shifu.shifu.combo;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Created by zhanhu on 12/9/16.
 */
public class ColumnFile {

    private static Logger LOG = LoggerFactory.getLogger(ColumnFile.class);

    private String filePath;
    private FileType fileType;
    private String delimiter;
    private String[] selectedVars;
    private Map<String, String> varsMapping;

    public ColumnFile(String filePath,
                      FileType fileType,
                      String delimiter,
                      String[] selectedVars,
                      Map<String, String> varsMapping) {
        this.filePath = filePath;
        this.fileType = fileType;
        this.delimiter = delimiter;
        this.selectedVars = selectedVars;
        this.varsMapping = varsMapping;
    }

    public Map<String, String> getVarsMapping() {
        return varsMapping;
    }

    public String getFilePath() {
        return filePath;
    }

    public FileType getFileType() {
        return fileType;
    }

    public String getDelimiter() {
        return delimiter;
    }

    public String[] getSelectedVars() {
        return selectedVars;
    }

    public List<String> getOutputVarNames() {
        List<String> outputVarNames = new ArrayList<String>();
        for ( String var : selectedVars ) {
            if ( this.varsMapping.containsKey(var) ) {
                outputVarNames.add(this.varsMapping.get(var));
            } else {
                outputVarNames.add(var);
            }
        }
        return outputVarNames;
    }

    public String genFieldSelector() {
        List<String> fields = new ArrayList<String>();
        for ( String var : selectedVars ) {
            if ( this.varsMapping.containsKey(var) ) {
                fields.add(var + " as " + this.varsMapping.get(var));
            } else {
                fields.add(var);
            }
        }
        return StringUtils.join(fields, ",");
    }

    public Map<String,List<String>> loadSelectedData(String keyName) {
        LOG.info("Load data from {}:{} by key {}.", fileType, filePath, keyName);

        Map<String,List<String>> selectedData = new HashMap<String, List<String>>();

        if ( FileType.CSV.equals(fileType) ) {
            CvsFile cvsFile = new CvsFile(filePath, delimiter);
            for ( Map<String, String> records : cvsFile ) {
                String key = records.get(keyName);
                if ( key != null ) {
                    List<String> vars = new ArrayList<String>();
                    for ( String varName : selectedVars ) {
                        vars.add(records.get(varName));
                    }
                    selectedData.put(key, vars);
                }

            }
        }

        return selectedData;
    }

    public boolean hasSelectedVar(String uidColumnName) {
        return ArrayUtils.contains(selectedVars, uidColumnName);
    }

    public static enum FileType {
        CSV, PIGSTORAGE
    }
}
