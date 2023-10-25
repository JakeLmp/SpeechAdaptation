# prevent interactive plot windows from opening
import matplotlib
matplotlib.use('Agg')

from . import main

if __name__ == "__main__":
    main()
