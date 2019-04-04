package ml.shifu.shifu.combo;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Implement CsvFile as iterator
 * Created by zhanhu on 12/10/16.
 */
public class CsvFile implements Iterable<Map<String, String>> {
    private static Logger LOG = LoggerFactory.getLogger(CsvFile.class);

    private String filePath;
    private String delimiter;
    private boolean removeNS;
    private CvsFileIterator iterator;

    public CsvFile(String filePath, String delimiter) {
        this.filePath = filePath;
        this.delimiter = delimiter;
        this.removeNS = false;
        this.iterator = new CvsFileIterator(filePath, delimiter, false);
    }

    public CsvFile(String filePath, String delimiter, boolean removeNS) {
        this.filePath = filePath;
        this.delimiter = delimiter;
        this.removeNS = removeNS;
        this.iterator = new CvsFileIterator(filePath, delimiter, removeNS);
    }

    public int getColumnOps(String column) {
        return ArrayUtils.indexOf(this.iterator.getHeader(), column);
    }

    public String[] getColumnNames() {
        return Arrays.copyOf(this.iterator.getHeader(), this.iterator.getHeader().length);
    }

    @Override
    public Iterator<Map<String, String>> iterator() {
        return new CvsFileIterator(filePath, delimiter, removeNS);
    }

    public static class CvsFileIterator implements Iterator<Map<String, String>> {
        private String nextLine;
        private BufferedReader reader;
        private boolean isFinished = false;
        private String[] header;
        private String delimiter;
        private boolean removeNS;

        public CvsFileIterator(String filePath, String delimiter, boolean removeNS) {
            LOG.info("Create csv file {}, with delimiter {}", filePath, delimiter);
            this.delimiter = delimiter;
            this.removeNS = removeNS;
            try {
                open(filePath);
            } catch (Exception e) {
                LOG.error("Fail to open file - {}.", filePath, e);
            }
        }

        private void open(String filePath) throws IOException {
            this.reader = new BufferedReader(new FileReader(filePath));
            String headerLine = reader.readLine();
            this.header = StringUtils.splitPreserveAllTokens(headerLine, delimiter);
            if ( this.removeNS ) {
                for ( int i = 0; i < this.header.length; i ++ ) {
                    this.header[i] = this.header[i].replaceAll(".*::", "");
                }
            }
        }

        @Override
        public boolean hasNext() {
            if(nextLine != null) {
                return true;
            } else if(isFinished) {
                return false;
            } else {
                try {
                    nextLine = reader.readLine();
                } catch (IOException e) {
                    nextLine = null;
                }

                if(nextLine == null) {
                    isFinished = true;
                    IOUtils.closeQuietly(reader);
                    return false;
                }

                return true;
            }
        }

        @Override
        public Map<String, String> next() {
            if(!hasNext()) {
                return null;
            }

            String[] vars = StringUtils.splitPreserveAllTokens(nextLine, delimiter);
            Map<String, String> varMap = new HashMap<String, String>();
            for(int i = 0; i < this.header.length; i++) {
                varMap.put(this.header[i], vars[i]);
            }

            nextLine = null;

            return varMap;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Remove is not supported.");
        }

        public String[] getHeader() {
            return this.header;
        }
    }
}
