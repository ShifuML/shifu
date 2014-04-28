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
package ml.shifu.shifu.fs;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import junit.framework.Assert;

import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.ShifuFileUtils;

import org.apache.commons.io.FileUtils;
import org.testng.annotations.Test;


/**
 * ShifuFileUtilsTest class
 * 
 */
public class ShifuFileUtilsTest {

	@Test
	public void getDataScannersTest() throws IOException{
		
		File file = new File("common-utils");
		if(!file.exists()){
			file.mkdirs();
		} else {
			FileUtils.deleteDirectory(file);
			file.mkdirs();
		}
		
		List<Scanner> list = ShifuFileUtils.getDataScanners(Arrays.asList(new String[]{"common-utils"}), SourceType.HDFS);
		
		Assert.assertTrue(list.size() == 0);
		
		file = new File("common-utils/part-0000");
		file.createNewFile();
		
		list = ShifuFileUtils.getDataScanners(Arrays.asList(new String[]{"common-utils"}), SourceType.HDFS);
		
		Assert.assertTrue(list.size() == 1);
		
		for(Scanner scanner:list){
			scanner.close();
		}
		
		FileUtils.deleteDirectory(new File("common-utils"));
	}
	
	@Test
	public void getReaderTest() throws IOException {
		BufferedReader reader = ShifuFileUtils.getReader("src/test/resources/example/wdbc/wdbcDataSet/wdbc.eval",
				SourceType.HDFS);

		Assert.assertEquals("84300903,M,19.69,21.25,130,1203,0.1096,0.1599,0.1974,0.1279,0.2069,0.05999,0.7456,0.7869,4.585,94.03,0.00615,0.04006,0.03832,0.02058,0.0225,0.004571,23.57,25.53,152.5,1709,0.1444,0.4245,0.4504,0.243,0.3613,0.08758", reader.readLine());

		reader.close();
	}
	
	@Test
	public void copyDataTest() throws IOException{
		File file = new File("common-utils/from_data");
		if(!file.exists()){
			file.mkdirs();
		}
		
		ShifuFileUtils.copy("common-utils/from_data", "common-utils/to_data", SourceType.LOCAL);
		
		file = new File("common-utils/to_data");
		
		Assert.assertTrue(file.exists());
	}
	
	@Test
	public void testIsFileExists() throws IOException {
		Assert.assertTrue(ShifuFileUtils.isFileExists("src/test/resources/example/wdbc/wdbcDataSet", SourceType.LOCAL));
		Assert.assertTrue(ShifuFileUtils.isFileExists("src\\test\\resources\\example\\wdbc\\wdbcDataSet", SourceType.LOCAL));
		Assert.assertTrue(ShifuFileUtils.isFileExists("src/test/resources/example/wdbc/wdbcDataSet/wdbc.{data,eval,header,train}", SourceType.LOCAL));
		Assert.assertTrue(ShifuFileUtils.isFileExists("src/test/resources/example/wdbc/wdbcDataSet/wdbc.{not-exists,eval,header,train}", SourceType.LOCAL));
		Assert.assertFalse(ShifuFileUtils.isFileExists("src/test/resources/example/wdbc/wdbcDataSet/wdbc.{not-exists,not-existsa,not-existsb}", SourceType.LOCAL));
	}
	
	@Test
	public void testExpandPath() throws IOException {
		File testDir = new File("test-dir");
		testDir.mkdir();
		
		File tmp0 = new File("test-dir/tmp0");
		File tmp1 = new File("test-dir/tmp1");
		File tmp2 = new File("test-dir/tmp2");
		
		tmp0.createNewFile();
		tmp1.createNewFile();
		tmp2.createNewFile();
		
		List<String> filePathList = ShifuFileUtils.expandPath("test-dir/tmp[0-2]", SourceType.LOCAL);
		Assert.assertEquals(3, filePathList.size());
		
		filePathList = ShifuFileUtils.expandPath("test-dir/{tmp0,tmp2}", SourceType.LOCAL);
		Assert.assertEquals(2, filePathList.size());
		Assert.assertTrue(filePathList.get(0).contains("test-dir/tmp0"));
		
		filePathList = ShifuFileUtils.expandPath("*-dir/*", SourceType.LOCAL);
		Assert.assertEquals(3, filePathList.size());
		Assert.assertTrue(filePathList.get(0).contains("test-dir/tmp0"));
		
		filePathList = ShifuFileUtils.expandPath("~", SourceType.LOCAL);
		Assert.assertEquals(0, filePathList.size());
		
		FileUtils.deleteDirectory(testDir);
	}
}
