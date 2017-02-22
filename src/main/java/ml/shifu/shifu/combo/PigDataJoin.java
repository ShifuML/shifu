package ml.shifu.shifu.combo;

import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.pig.PigExecutor;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhanhu on 12/13/16.
 */
public class PigDataJoin {
    private static Logger LOG = LoggerFactory.getLogger(PigDataJoin.class);
    private static final String DATA_PREFIX = "data";

    public void join(String uidColumnName, String outputPath, List<ColumnFile> columnFileList) throws IOException {
        String pigCode = genPigJoinCode(uidColumnName, outputPath, columnFileList);
        LOG.debug("\n" + pigCode);

        // Run pig code to merge data
        PigExecutor.getExecutor().submitJob(RawSourceData.SourceType.HDFS, pigCode);
    }

    /*
     * Generate pig code for data merge
     * 
     * @param uidColumnName - the column to join
     * 
     * @param outputPath - the output path for joined file
     * 
     * @param columnFileList
     * 
     * @return
     * 
     * @throws IOException
     */
    public String genPigJoinCode(String uidColumnName, String outputPath, List<ColumnFile> columnFileList)
            throws IOException {
        ByteArrayOutputStream byos = new ByteArrayOutputStream();
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(byos));

        try {
            List<String> relations = new ArrayList<String>();

            int i = 0;
            for(ColumnFile columnFile: columnFileList) {
                String relation = DATA_PREFIX + (i++);
                writeLine(writer, relation + " = load '" + columnFile.getFilePath()
                        + "' using PigStorage('|', '-schema');");
                if(columnFile.hasSelectedVar(uidColumnName)) {
                    writeLine(writer,
                            relation + " = foreach " + relation + " generate " + columnFile.genFieldSelector() + ";");
                } else {
                    writeLine(writer, relation + " = foreach " + relation + " generate " + uidColumnName + " as "
                            + uidColumnName + ", " + columnFile.genFieldSelector() + ";");
                }
                relations.add(relation);
            }

            writeLine(writer, "result = group " + genGroupByClauses(relations, uidColumnName) + ";");
            writeLine(writer, "result = foreach result " + genLimitClauses(relations, uidColumnName) + ";");
            writeLine(writer, "result = filter result by " + genFilterSizeClauses(relations) + ";");
            writeLine(writer, "result = foreach result generate " + genFlattenClauses(relations) + ";");
            writeLine(writer, "result = foreach result generate " + genRenameClauses(columnFileList, relations) + ";");
            writeLine(writer, "rmf " + outputPath + ";");
            writeLine(writer, "store result into '" + outputPath + "' using PigStorage('|', '-schema');");
        } catch (IOException e) {
            LOG.error("Fail to generate pig code for data merge.", e);
            throw e;
        } finally {
            IOUtils.closeQuietly(writer);
            IOUtils.closeQuietly(byos);
        }

        return byos.toString();
    }

    /*
     * Generate filter by size clauses
     */
    private String genFilterSizeClauses(List<String> relations) {
        List<String> filterByClauses = new ArrayList<String>();
        for(String relation: relations) {
            filterByClauses.add("SIZE(" + relation + ") == 1");
        }
        return StringUtils.join(filterByClauses, " and ");
    }

    /*
     * Generate group by clauses
     * 
     * @param relations - Relation list
     * 
     * @param uidColumnName - join columnName
     * 
     * @return - pig group list
     */
    private String genGroupByClauses(List<String> relations, String uidColumnName) {
        List<String> groupByClauses = new ArrayList<String>();
        for(String relation: relations) {
            groupByClauses.add(relation + " by " + uidColumnName);
        }
        return StringUtils.join(groupByClauses, ",");
    }

    /*
     * Generate limit 1 clause after group-by
     * 
     * @param relations - Relation list
     * 
     * @param uidColumnName - join column Name
     * 
     * @return pig limit list
     */
    private String genLimitClauses(List<String> relations, String uidColumnName) {
        List<String> limitsClauses = new ArrayList<String>();
        for(String relation: relations) {
            limitsClauses.add(relation + " = limit " + relation + " 1");
        }

        StringBuffer buf = new StringBuffer();
        buf.append("generate group," + StringUtils.join(relations, ","));
        limitsClauses.add(buf.toString());

        return "{ " + StringUtils.join(limitsClauses, ";") + "; }";
    }

    /*
     * Generate flatten clauses
     * 
     * @param relations
     * 
     * @return - pig flatten list
     */
    private String genFlattenClauses(List<String> relations) {
        List<String> flattenClauses = new ArrayList<String>();
        for(String relation: relations) {
            flattenClauses.add("FLATTEN(" + relation + ")");
        }
        return StringUtils.join(flattenClauses, ",");
    }

    /*
     * Generate fields rename clauses
     * 
     * @param columnFileList
     * 
     * @param relations
     * 
     * @return - pig rename(as) list
     */
    private String genRenameClauses(List<ColumnFile> columnFileList, List<String> relations) {
        List<String> renameClauses = new ArrayList<String>();
        for(int i = 0; i < columnFileList.size(); i++) {
            ColumnFile columnFile = columnFileList.get(i);
            String relation = relations.get(i);

            List<String> outputVars = columnFile.getOutputVarNames();
            for(String var: outputVars) {
                renameClauses.add(relation + "::" + var + " as " + var);
            }
        }

        return StringUtils.join(renameClauses, ",");
    }

    /*
     * Write line with "\n"
     * 
     * @param writer
     * 
     * @param line
     * 
     * @throws IOException
     */
    private void writeLine(BufferedWriter writer, String line) throws IOException {
        writer.write(line);
        writer.newLine();
    }
}
