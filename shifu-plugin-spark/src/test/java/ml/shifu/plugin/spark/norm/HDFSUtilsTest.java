package ml.shifu.plugin.spark.norm;

import java.io.IOException;

import ml.shifu.plugin.spark.norm.HDFSFileUtils;

import org.apache.hadoop.fs.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

public class HDFSUtilsTest {

    private String pathHDFSConf = "/usr/local/hadoop/etc/hadoop";
    private HDFSFileUtils hdfsUtils;

    public HDFSUtilsTest() throws IOException {
        this.hdfsUtils = new HDFSFileUtils(pathHDFSConf);

    }

    /*
     * @Test public void createFileTest() throws IOException { // local file
     * this.hdfsUtils.createEmptyFile("file:///Users/apalnitkar/temp_file");
     * this.hdfsUtils.createEmptyFile("hdfs:///temp_file"); }
     */

    /*
     * @Test public void deleteTest() throws IOException { // create file in
     * local FS and hadoop FS Path local= "file://Users/apalnitkar/temp_file";
     * Path hdfs= "hdfs:///temp_file"; this.hdfsUtils.create(local);
     * this.hdfsUtils.createEmptyFile(hdfs); this.hdfsUtils.delete();
     * this.hdfsUtils.delete();
     * 
     * }
     */

    /*
     * @Test public void uploadToHDFSIfLocalTest() throws Exception {
     * this.hdfsUtils.uploadToHDFSIfLocal("~/temp_file", "ml/shifu/norm/tmp");
     * this.hdfsUtils.uploadToHDFSIfLocal("hdfs:///temp_file",
     * "ml/shifu/norm/tmp");
     * 
     * }
     */
}
