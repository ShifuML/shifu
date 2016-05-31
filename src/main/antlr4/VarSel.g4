grammar VarSel;

varsel : expr ('and'  expr)+;
expr : (ID opt NUM) | ID ;
ID: [a-zA-Z]+;
opt: '>' | '<' | '>=' | '<=' | '=' ;
NUM : '-'? INT '.' [0-9] + EXP? | '-'? INT EXP | '-'? INT;
fragment INT
   : '0' | [1-9] [0-9]*
   ;
fragment EXP
   : [Ee] [+\-]? INT
   ;
