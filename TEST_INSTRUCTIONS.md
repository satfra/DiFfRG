To check if DiFfRG works as intended, you may

1. Build and run all the Examples in the Examples/ directory:
    - In each sub-directory, there is a build.sh file, which you can run as 
      ```
      $ bash build.sh -j8``
      ```
    - After building, change directories into the newly created build/ folder, and simply run the executable present, e.g.
        - Yang-Mills for `Examples/Yang-Mills/SP` and `Examples/Yang-Mills/Full`
        - `CG`, `LDG` for `Examples/ONfiniteT`. `dDG` gets intentionally stuck during the time-stepping
        - `QuarkMeson` for `Examples/QuarkMesonLPAprime/`
        - `FourFermi` for `Examples/FourFermi/`
2. Run the tests:
    ```
    $ bash run_tests.sh -j8
    ```
