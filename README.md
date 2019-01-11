It is aimed to analyze the mathematical equations mentioned below. For this, the java language and javacc (the parser generator developed for java language) were used.

As examples of mathematical iterative equations, equations expressing the numbers of fibonacci can be expressed as follows.

input.txt file contains the equations and looks like this:
```
F(0) = 0
F(1) = 1
F(x) = F(x-1) + F(x-2)
```
To compile the project;
```
javacc lang.jj
javac *.java
```
To execute;
```
java EvalVisitor < prog.txt
```
