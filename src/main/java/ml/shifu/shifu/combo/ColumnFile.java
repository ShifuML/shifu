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

    /**
     * file location
     */
    private String filePath;

    /**
     * file type, CSV format or PigStorage file
     */
    private FileType fileType;

    /**
     * file delimiter
     */
    private String delimiter;

    /**
     * variables list that will be selected
     */
    private String[] selectedVars;

    /**
     * variable mapping, from variable name to output variable name
     */
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

    /**
     * generate output variables after mapping
     * @return list output variable names
     */
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

    /**
     * generate the fields projector for selected variables
     * @return pig field selector
     */
    public String genFieldSelector() {
        List<String> fields = new ArrayList<String>();
        for ( String var : selectedVars ) {
            if ( this.varsMapping.containsKey(var) ) {
                fields.add(var + " as " + this.varsMapping.get(var));
            } else {
                fields.add(var + " as " + var);
            }
        }
        return StringUtils.join(fields, ",");
    }

    /**
     * Load data into memory, only selected data.
     * The output format is &lt;key, selected-variables&gt;
     * @param keyName the key field for records
     * @return - &lt;key, selected-variables&gt;
     */
    public Map<String,List<String>> loadSelectedData(String keyName) {
        LOG.info("Load data from {}:{} by key {}.", fileType, filePath, keyName);

        Map<String,List<String>> selectedData = new HashMap<String, List<String>>();

        if ( FileType.CSV.equals(fileType) ) {
            CsvFile cvsFile = new CsvFile(filePath, delimiter);
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

    /**
     * Check the selected variables contain some variable
     * @param varName - variable name to check
     * @return - whether the variable selected or not
     */
    public boolean hasSelectedVar(String varName) {
        return ArrayUtils.contains(selectedVars, varName);
    }

    public static enum FileType {
        CSV, PIGSTORAGE
    }
}
