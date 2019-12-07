# Configure GUI
from ranking.gooey import GooeyParser

parser = GooeyParser()
parser.add_argument('dsn', metavar='DSN')
parser.add_argument('clientname', metavar='Client Name',)
parser.add_argument('outputdir', metavar='Output Folder', widget='DirChooser')
grp = parser.add_mutually_exclusive_group(required=True, gooey_options={'show_border': True})
grp.add_argument('--database', metavar='SPR Client')
grp.add_argument('--filename', metavar='NonSPR Client - Rolled Up', widget='FileChooser',
                 help='Columns = Supplier, Year, Total_Invoice_Amount, Total_Invoice_Count')
grp.add_argument('--filename2', metavar='NonSPR Client - Raw', widget='FileChooser',
                 help='Columns = Vendor Name, Invoice Date, Gross Invoice Amount')
