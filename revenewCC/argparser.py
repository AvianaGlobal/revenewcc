# Configure GUI
from gooey import GooeyParser

parser = GooeyParser()
parser.add_argument('dsn',
                    metavar='ODBC Data Source Name (DSN)',
                    help='Please enter the DSN below',
                    action='store', )
parser.add_argument('clientname', metavar='CC Client Name',
                    help='Please enter the client\'s name below',
                    action='store', )
parser.add_argument('outputdir', metavar='Output Data Folder',
                    help='Please select a target directory',
                    action='store',
                    widget='DirChooser')
grp = parser.add_mutually_exclusive_group(required=True,
                                          gooey_options={'show_border': True})
grp.add_argument('--database',
                 metavar='SPR Client',
                 widget='TextField',
                 help='Please enter the client database name',
                 action='store', )
grp.add_argument('--filename',
                 metavar='NonSPR Client - Rolled Up',
                 widget='FileChooser',
                 help='CSV file '
                      'Columns = Client, Supplier, Year, Total_Invoice_Amount, Total_Invoice_Count',
                 action='store', )
grp.add_argument('--filename2',
                 metavar='NonSPR Client - Raw',
                 widget='FileChooser',
                 help='CSV file '
                      'Columns: Vendor Name, Invoice Date, Gross Invoice Amount',
                 action='store', )
