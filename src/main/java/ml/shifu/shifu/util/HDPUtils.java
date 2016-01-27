/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.util;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.net.URLDecoder;
import java.util.Enumeration;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.DistributedFileSystem;
import org.apache.hadoop.io.IOUtils;
import org.apache.pig.impl.PigContext;
import org.apache.pig.impl.util.JarManager;

/**
 * For HortonWorks HDP 2.2.4 two issues not caomptible with other hadoop 2.x versions.
 * 
 * <p>
 * Two major issues include:
 * <ul>
 * <li>1. Cannot find hdp.version on the fly, we need find the version firstly and we need find such number from
 * dependency jar and set it on Hadoop Configuration object.</li>
 * <li>2. HADOOP_CONF_DIR is not set as classpath in classpath of every container, which make Shifu UDF and
 * mapper/reducer cannot read configuration from HDFS. To solve it we have to find such configuration files like
 * 'yarn-site.xml', 'core-site.xml', 'hdfs-site.xml' and 'mapred-site.xml' on client class path and ship them to
 * container classpath as distributed cache files or jars.</li>
 * </ul>
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class HDPUtils {

    /**
     * A ugly method to retrieve hdp version like 2.2.4.6633 from hadoop-hdfs jar name. The jar name is like
     * 'hadoop-hdfs-2.6.0.2.2.4.6633.jar'.
     */
    public static String getHdpVersionForHDP224() {
        String hdfsJarWithVersion = JarManager.findContainingJar(DistributedFileSystem.class);
        String hdpVersion = "";
        if(hdfsJarWithVersion != null) {
            if(hdfsJarWithVersion.contains(File.separator)) {
                hdfsJarWithVersion = hdfsJarWithVersion.substring(hdfsJarWithVersion.lastIndexOf(File.separator) + 1);
            }
            hdfsJarWithVersion = hdfsJarWithVersion.replace("hadoop-hdfs-", "");
            hdfsJarWithVersion = hdfsJarWithVersion.replace(".jar", "");
            String[] splits = hdfsJarWithVersion.split("\\.");
            if(splits.length > 2) {
                for(int i = 3; i < splits.length; i++) {
                    if(i == splits.length - 1) {
                        hdpVersion += splits[i];
                    } else {
                        hdpVersion += splits[i] + ".";
                    }
                }
            }
        }
        return hdpVersion;
    }

    /**
     * Add a hdfs file to classpath of one container.
     */
    @SuppressWarnings("deprecation")
    public static void addFileToClassPath(String file, Configuration conf) throws IOException {
        Path pathInHDFS = shipToHDFS(conf, file);
        org.apache.hadoop.filecache.DistributedCache.addFileToClassPath(pathInHDFS, conf, FileSystem.get(conf));
    }

    /**
     * Copy local file to HDFS. This is used to set as classpath by distributed cache.
     */
    private static Path shipToHDFS(Configuration conf, String fileName) throws IOException {
        Path dst = new Path(File.separator + "tmp" + File.separator + System.getProperty("user.name") + "_"
                + System.currentTimeMillis(), fileName.substring(fileName.lastIndexOf(File.separator) + 1));
        FileSystem fs = dst.getFileSystem(conf);
        OutputStream os = null;
        InputStream is = null;
        try {
            is = FileSystem.getLocal(conf).open(new Path(fileName));
            os = fs.create(dst);
            IOUtils.copyBytes(is, os, 4096, true);
        } finally {
            org.apache.commons.io.IOUtils.closeQuietly(is);
            // IOUtils should not close stream to HDFS quietly
            if(os != null) {
                os.close();
            }
        }
        return dst;
    }

    /**
     * Find a real file that contains file name in class path.
     * 
     * @param file
     *            name
     * @return real file name
     */
    public static String findContainingFile(String fileName) {
        ClassLoader loader = PigContext.getClassLoader();
        try {
            Enumeration<URL> itr = null;
            // Try to find the class in registered jars
            if(loader instanceof URLClassLoader) {
                itr = ((URLClassLoader) loader).findResources(fileName);
            }
            // Try system classloader if not URLClassLoader or no resources found in URLClassLoader
            if(itr == null || !itr.hasMoreElements()) {
                itr = loader.getResources(fileName);
            }
            for(; itr.hasMoreElements();) {
                URL url = (URL) itr.nextElement();
                if("file".equals(url.getProtocol())) {
                    String toReturn = url.getPath();
                    if(toReturn.startsWith("file:")) {
                        toReturn = toReturn.substring("file:".length());
                    }
                    // URLDecoder is a misnamed class, since it actually decodes
                    // x-www-form-urlencoded MIME type rather than actual
                    // URL encoding (which the file path has). Therefore it would
                    // decode +s to ' 's which is incorrect (spaces are actually
                    // either unencoded or encoded as "%20"). Replace +s first, so
                    // that they are kept sacred during the decoding process.
                    toReturn = toReturn.replaceAll("\\+", "%2B");
                    toReturn = URLDecoder.decode(toReturn, "UTF-8");
                    return toReturn.replaceAll("!.*$", "");
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return null;
    }
}
