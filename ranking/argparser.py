# Configure GUI
from ranking.gooey import GooeyParser

parser = GooeyParser()
parser.add_argument('clientname', metavar='Client Name')
parser.add_argument('outputdir', metavar='Output Folder', widget='DirChooser')

spr_grp = parser.add_group('Option 1: SPR Client')
spr_grp.add_argument('--dsn', metavar='DSN')
spr_grp.add_argument('--database', metavar='Database')
spr_grp.add_argument('--username', metavar='Username')
spr_grp.add_argument('--password', metavar='Password', widget='PasswordField')

nonspr_grp = parser.add_group('Option 2: Non-SPR Client')
nonspr_grp.add_argument('--filename', metavar='Rolled Up', widget='FileChooser',
                 help='Columns: Supplier, Year, Total_Invoice_Amount, Total_Invoice_Count')
nonspr_grp.add_argument('--filename2', metavar='Raw', widget='FileChooser',
                 help='Columns: Vendor Name, Invoice Date, Gross Invoice Amount')
