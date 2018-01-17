package ml.shifu.shifu.column;

import java.util.Arrays;

import org.apache.commons.lang.StringUtils;

/**
 * Created by zhanhu on 3/23/17.
 */
public class NSColumn {

    private static final String NS_DELIMITER = "::";

    private String fullColumnName;
    private String[] nameIdentifiers;

    public NSColumn(String fullColumnName) {
        this.fullColumnName = fullColumnName;
        if ( StringUtils.isNotBlank(this.fullColumnName) ) {
            this.nameIdentifiers = StringUtils.split(this.fullColumnName, NS_DELIMITER, -1);
        }
    }

    public String getFullColumnName() {
        return fullColumnName;
    }

    public String getSimpleName() {
        return (this.nameIdentifiers == null ? null : this.nameIdentifiers[this.nameIdentifiers.length -1]);
    }

    public String[] getNameIdentifiers() {
        return nameIdentifiers;
    }

    @Override
    public int hashCode() {
        return ( this.nameIdentifiers != null && this.nameIdentifiers.length > 0 ) ?
            this.nameIdentifiers[this.nameIdentifiers.length - 1].hashCode() : 0;
    }

    @Override
    public boolean equals(Object obj) {
        if ( obj == null || !(obj instanceof NSColumn) ) {
            return false;
        }

        if ( obj == this ) {
            return true;
        }

        NSColumn nsc = (NSColumn) obj;
        if ( StringUtils.equals(this.fullColumnName, nsc.fullColumnName) ) {
            return true;
        } else {
            boolean equal = false;
            if ( this.nameIdentifiers != null && nsc.nameIdentifiers != null ) {
                if (this.nameIdentifiers.length == 0 && nsc.nameIdentifiers.length == 0) {
                    equal = true;
                } else if (this.nameIdentifiers.length * nsc.nameIdentifiers.length == 0) {
                    equal = false;
                } else {
                    int len = Math.min(this.nameIdentifiers.length, nsc.nameIdentifiers.length);
                    int k;
                    for ( k = 1; k < len + 1; k ++) {
                        if (!StringUtils.equals(this.nameIdentifiers[this.nameIdentifiers.length - k],
                                nsc.nameIdentifiers[nsc.nameIdentifiers.length - k])) {
                            break;
                        }
                    }
                    equal = (k == (len +1 ));
                }
            }
            return equal;
        }
    }

    /* (non-Javadoc)
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        return "NSColumn [fullColumnName=" + fullColumnName + ", nameIdentifiers=" + Arrays.toString(nameIdentifiers)
                + "]";
    }
}
