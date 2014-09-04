/**
 * Copyright [2012-2014] eBay Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.plugin.spark.utils;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

import ml.shifu.core.request.Request;
import ml.shifu.core.util.JSONUtils;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.utils.HDFSFileUtils;

public class HDFSUtilsTest {
    HDFSFileUtils hdfsUtils;
    
    @BeforeTest
    public void instantiate() throws IOException, IllegalArgumentException, URISyntaxException {
        Request req=  JSONUtils.readValue(new File("src/test/resources/stats/spark_stats.json"), Request.class); 
        Params params= req.getProcessor().getParams();
        String hdfsConf= params.get("pathHDFSConf", "/usr/lib/hadoop/etc/hadoop").toString();
        hdfsUtils= new HDFSFileUtils(hdfsConf);
    }
    
    @Test
    public void testDelete() throws IOException, IllegalArgumentException, URISyntaxException {
        // create a file in local filesystem
        File testFile= new File("src/test/resources/stats/HDFSTest/dummy");
        FileUtils.touch(testFile); 
        hdfsUtils.delete(testFile.toURI().toString());
        Assert.assertFalse(hdfsUtils.exists(testFile.getPath()));
        FileUtils.deleteQuietly(testFile);
        
        // test on HDFS
        Path testPath= new Path(hdfsUtils.relativeToFullHDFSPath("ml/shifu/test/dummy"));
        FileSystem hdfs= FileSystem.get(hdfsUtils.getHDFSConf());
        hdfs.createNewFile(testPath);
        hdfs.close();
        hdfsUtils.delete(testPath.toString());
        Assert.assertFalse(hdfsUtils.exists(testPath.toString()));
        
        FileSystem hdfs1= FileSystem.get(hdfsUtils.getHDFSConf());
        hdfs1.delete(testPath, true);
        hdfs1.close();
    }
    
    
    @Test
    public void uploadToHDFSIfLocalTest() throws Exception {
        File testFile= new File("src/test/resources/stats/HDFSTest/dummy");
        FileUtils.touch(testFile);
        hdfsUtils.uploadToHDFSIfLocal(testFile.getAbsolutePath(), "ml/shifu/test");
        Assert.assertTrue(hdfsUtils.exists(hdfsUtils.relativeToFullHDFSPath("ml/shifu/test/dummy")));
        hdfsUtils.delete(hdfsUtils.relativeToFullHDFSPath("ml/shifu/test/dummy"));
        
        // check with hdfs file
        String hdfsPath= hdfsUtils.relativeToFullHDFSPath("ml/shifu/test1/dummy");
        hdfsUtils.createEmptyFile(hdfsPath);
        hdfsUtils.uploadToHDFSIfLocal(hdfsPath, "ml/shifu/test");
        
        Assert.assertFalse(hdfsUtils.exists(hdfsUtils.relativeToFullHDFSPath("ml/shifu/test/dummy")));
        hdfsUtils.delete(hdfsPath);
    }
    
    @Test 
    public void fullDefaultLocalTest() throws IOException, URISyntaxException {
        Assert.assertTrue(hdfsUtils.fullDefaultLocal("/home/user").equals("file:///home/user"));        
    }
}
