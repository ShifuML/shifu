/*
 * Copyright [2012-2014] PayPal Software Foundation
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

import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * {@link Environment} is used to store common env like 'SHIFU_HOME' and return to user by calling
 * {@link #getProperty(String)} method
 */
public class Environment {

    private static final String OS_NAME = "os.name";
    private static final String UNIX_SUFFIX_1 = "nix";
    private static final String UNIX_SUFFIX_2 = "nux";
    private static final String UNIX_SUFFIX_3 = "aix";
    private static final String UNIX_SUFFIX_4 = "mac os";
    private static final String WIN_PREFIX = "win";
    private static final String USER_NAME = "user.name";
    private static final String USER = "USER";

    public static final String SHIFU_HOME = "SHIFU_HOME";
    public static final String SYSTEM_USER = "SYSTEM_USER";

    public static final String ZOO_KEEPER_SERVERS = "zookeeperServers";
    public static final String HADOOP_NUM_PARALLEL = "hadoopNumParallel";
    public static final String LOCAL_NUM_PARALLEL = "localNumParallel";
    public static final String RECORD_CNT_PER_MESSAGE = "recordCntPerMessage";
    public static final String HADOOP_JOB_QUEUE = "hadoopJobQueue";

    public static final String VAR_SEL_MASTER_CONDUCTOR = "varselectMasterConductor";
    public static final String VAR_SEL_WORKER_CONDUCTOR = "varselectWorkerConductor";

    private static Logger logger = LoggerFactory.getLogger(Environment.class);
    private static Properties properties = new Properties();

    static {
        String shifuHomePath = ((System.getenv(SHIFU_HOME) == null) ? System.getProperty(SHIFU_HOME) : System
                .getenv(SHIFU_HOME));
        properties.put(SHIFU_HOME, ((shifuHomePath == null) ? "" : shifuHomePath));

        try {
            loadShifuConfig();
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_SHIFU_CONFIG, e);
        }

        if(properties.size() == 1) {
            logger.warn("No shifuconfig is found or there is no content in it");
        }

        String osName = System.getProperty(OS_NAME).toLowerCase();
        if(isUnix(osName)) {
            properties.put(SYSTEM_USER, System.getenv(USER));
        } else if(isWindows(osName)) {
            properties.put(SYSTEM_USER, System.getProperty(USER_NAME));
        }
    }

    /*
     * Load properties from
     * 1. ${SHIFU_HOME}/conf/shifuconfig
     * 2. /etc/shifuconfig
     * 3. ~/.shifuconfig
     * 
     * Provide function to reload
     */
    public static void loadShifuConfig() throws IOException {
        // check ${SHIFU_HOME}/conf/shifuconfig, if exists, load it
        loadProperties(properties, getProperty(Environment.SHIFU_HOME) + File.separator + "conf" + File.separator
                + "shifuconfig");

        loadProperties(properties, getProperty(Environment.SHIFU_HOME) + File.separator + "conf" + File.separator
                + "shifu.config");

        loadProperties(properties, getProperty(Environment.SHIFU_HOME) + File.separator + "shifu.config");

        // check /etc/shifuconfig, if exists, load it
        loadProperties(properties, File.separator + "etc" + File.separator + "shifuconfig");

        // check /<user-home>/.shifuconfig, if exists, load it
        String userHome = System.getProperty("user.home");
        loadProperties(properties, userHome + File.separator + ".shifuconfig");
    }

    /*
     * Get global property by property name
     */
    public static String getProperty(String propertyName) {
        return properties.getProperty(propertyName);
    }

    public static void setProperty(String propertyName, String propertyValue) {
        properties.put(propertyName, propertyValue);
    }

    /*
     * Get property, if null return default value
     */
    public static String getProperty(String propertyName, String defValue) {
        String propertyValue = getProperty(propertyName);
        return (propertyValue == null) ? defValue : propertyValue;
    }

    /*
     * Get property as Integer value
     */
    public static Integer getInt(String propertyName) {
        String propertyValue = getProperty(propertyName);
        return (propertyValue == null) ? null : Integer.valueOf(propertyValue);
    }

    /*
     * Get property as Integer value, if null return default value
     */
    public static Integer getInt(String propertyName, Integer defValue) {
        String propertyValue = getProperty(propertyName);
        return (propertyValue == null) ? defValue : Integer.valueOf(propertyValue);
    }

    /*
     * Get property as Boolean value, if null return default value
    */
    public static Boolean getBoolean(String propertyName, Boolean defValue) {
        String propertyValue = getProperty(propertyName);
        return StringUtils.isBlank(propertyValue) ? defValue : Boolean.valueOf(propertyValue);
    }

    /*
     * Get property as Long value
     */
    public static Long getLong(String propertyName) {
        String propertyValue = getProperty(propertyName);
        return (propertyValue == null) ? null : Long.valueOf(propertyValue);
    }

    /*
     * Get property as Integer value, if null return default value
     */
    public static Long getLong(String propertyName, Long defValue) {
        String propertyValue = getProperty(propertyName);
        return (propertyValue == null) ? defValue : Long.valueOf(propertyValue);
    }

    /**
     * Check the system type is Windows or not
     * 
     * @param osName
     *            osName from env
     * @return true if it is windows, or return false
     */
    private static boolean isWindows(String osName) {
        return (osName.indexOf(WIN_PREFIX) >= 0);
    }

    /**
     * Check the system type is Unix or not
     * 
     * @param osName
     *            osName from env
     * @return true if it is windows, or return false
     */
    private static boolean isUnix(String osName) {
        return (osName.indexOf(UNIX_SUFFIX_1) >= 0 || osName.indexOf(UNIX_SUFFIX_2) >= 0 || osName
                .indexOf(UNIX_SUFFIX_3) > 0) || osName.indexOf(UNIX_SUFFIX_4) >= 0;
    }

    /*
     * Load shifuconfig into properties
     */
    private static void loadProperties(Properties props, String fileName) throws IOException {
        File configFile = new File(fileName);
        if(!configFile.exists()) {
            return;
        }

        FileInputStream inStream = null;
        try {
            inStream = new FileInputStream(configFile);
            props.load(inStream);
        } finally {
            IOUtils.closeQuietly(inStream);
        }
    }

    /*
     * Get copied properties to make others can read them and send useful info to others like guagua framework.
     */
    public static Properties getProperties() {
        return Environment.properties;
    }

}